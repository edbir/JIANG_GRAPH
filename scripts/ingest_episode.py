"""
Ingest Episode — End-to-End Pipeline for a Single Episode.
============================================================

THIS IS THE SCRIPT YOU RUN to process a new episode into the knowledge graph.

WHAT IT DOES (in order):
    1. Read the transcript file
    2. Clean the text (remove fillers, artifacts, normalize)
    3. Chunk the text (LLM segmentation + token-bounded split)
    4. Extract knowledge (entities, relationships, claims, coreferences)
    5. Resolve entities (deduplicate across the episode)
    6. Upsert everything into Neo4j
    7. Generate and store embeddings
    8. (Optional) Re-run community detection

USAGE:
    python -m scripts.ingest_episode \
        --transcript data/transcripts/ep001.txt \
        --episode-id ep001 \
        --title "The Fall of the Roman Republic" \
        --number 1

    Or from Python:
        import asyncio
        from scripts.ingest_episode import ingest_episode
        asyncio.run(ingest_episode("data/transcripts/ep001.txt", "ep001", "Title", 1))

PREREQUISITES:
    - OpenAI API key in .env (OPENAI_API_KEY)
    - Neo4j running (docker compose up -d)
    - Neo4j password in .env (NEO4J_PASSWORD)
    - Schema created (run setup_schema() once — this script does it automatically)

COST:
    ~$0.08 per episode (extraction) + ~$0.01 (embeddings) = ~$0.09 total
    For the MVP (5 episodes): ~$0.45
"""

import asyncio
import argparse
from pathlib import Path

from openai import AsyncOpenAI

from src.config import settings
from src.preprocessing.text_cleaner import clean_transcript
from src.preprocessing.chunker import chunk_transcript, Segment
from src.extraction.extractor import process_episode
from src.extraction.resolver import (
    resolve_entities,
    EntityCandidate,
)
from src.graph.neo4j_client import GraphClient
from src.graph.upsert import upsert_episode_data, store_embedding


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================


async def generate_embeddings(
    openai_client: AsyncOpenAI,
    graph_client: GraphClient,
    chunks: list,
    extraction_results: dict,
    merge_map: dict[str, str],
) -> int:
    """
    Generate and store embeddings for all nodes that need them.
    
    Embeds:
    - Utterances (chunks): full text
    - Claims: "claim text | entities involved"
    - Entities: "name: description (aliases)"
    
    Community embeddings are handled separately in communities.py.
    
    Returns:
        Number of embeddings generated
    """
    count = 0
    texts_to_embed: list[tuple[str, str, str]] = []
    # Each tuple: (label, id_field_value, text_to_embed)

    # --- Utterance embeddings ---
    for chunk in chunks:
        texts_to_embed.append(("Utterance", chunk.chunk_id, chunk.text))

    # --- Claim embeddings ---
    claims_data = extraction_results.get("claims", {})
    for chunk_id, claim_result in claims_data.items():
        for claim in claim_result.claims:
            if claim.confidence >= 0.7:
                from src.graph.upsert import make_claim_id
                claim_id = make_claim_id(claim.text)
                # Embed: claim text + entities for richer semantic representation
                entities_str = ", ".join(claim.entities_involved)
                embed_text = f"{claim.text} | Entities: {entities_str}"
                texts_to_embed.append(("Claim", claim_id, embed_text))

    # --- Entity embeddings ---
    # Collect unique entities (after merge map applied)
    seen_entities: set[str] = set()
    entities_data = extraction_results.get("entities", {})
    for chunk_id, entity_result in entities_data.items():
        for entity in entity_result.entities:
            canonical = merge_map.get(entity.name, entity.name)
            if canonical in seen_entities:
                continue
            seen_entities.add(canonical)

            from src.graph.upsert import make_entity_id
            entity_id = make_entity_id(entity.entity_type.value, canonical)

            # Embed: "name: description (aliases)"
            aliases_str = ", ".join(entity.aliases) if entity.aliases else ""
            embed_text = f"{canonical}: {entity.description}"
            if aliases_str:
                embed_text += f" (also known as: {aliases_str})"

            # Determine label and id_field
            label_map = {
                "person": ("Person", "personId"),
                "organization": ("Organization", "orgId"),
                "topic": ("Topic", "topicId"),
                "event": ("Event", "eventId"),
            }
            label, id_field = label_map[entity.entity_type.value]
            texts_to_embed.append((f"{label}:{id_field}", entity_id, embed_text))

    # --- Batch embed (OpenAI supports up to 2048 inputs per call) ---
    BATCH_SIZE = 100
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        texts = [item[2] for item in batch]

        response = await openai_client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )

        # Store each embedding
        for j, embedding_data in enumerate(response.data):
            label_info, node_id, _ = batch[j]
            embedding = embedding_data.embedding

            # Parse label and id_field
            if ":" in label_info:
                label, id_field = label_info.split(":")
            else:
                label = label_info
                # Map label to id_field
                id_field_map = {
                    "Utterance": "utteranceId",
                    "Claim": "claimId",
                }
                id_field = id_field_map.get(label, "id")

            await store_embedding(graph_client, label, id_field, node_id, embedding)
            count += 1

    return count


# =============================================================================
# ENTITY CANDIDATE COLLECTION
# =============================================================================


def collect_entity_candidates(extraction_results: dict) -> list[EntityCandidate]:
    """
    Collect all extracted entities into EntityCandidate objects for resolution.
    
    Groups entities by name and aggregates their source chunks.
    """
    # Group by canonical name
    candidates_map: dict[str, EntityCandidate] = {}

    entities_data = extraction_results.get("entities", {})
    for chunk_id, entity_result in entities_data.items():
        for entity in entity_result.entities:
            if entity.name not in candidates_map:
                candidates_map[entity.name] = EntityCandidate(
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    description=entity.description,
                    aliases=list(entity.aliases),
                    source_chunks=[chunk_id],
                )
            else:
                # Entity seen before — accumulate info
                existing = candidates_map[entity.name]
                existing.source_chunks.append(chunk_id)
                # Keep longer description
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
                # Accumulate aliases
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)

    return list(candidates_map.values())


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================


async def ingest_episode(
    transcript_path: str,
    episode_id: str,
    title: str,
    episode_number: int,
    episode_context: str = "",
    run_communities: bool = False,
) -> dict:
    """
    Full end-to-end ingestion of a single episode.
    
    This is the main function. It orchestrates the entire pipeline:
    clean → chunk → extract → resolve → upsert → embed → (communities)
    
    Parameters:
        transcript_path: Path to the transcript text file
        episode_id:      Unique episode identifier (e.g., "ep001")
        title:           Episode title
        episode_number:  Episode number (integer)
        episode_context: Known subjects/focus for this episode (helps extraction)
        run_communities: Whether to re-run community detection after ingestion
    
    Returns:
        Dict with statistics from each step
    """
    stats: dict = {}

    # --- Initialize clients ---
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    # print(f"[ingest] Starting ingestion of {episode_id}: {title}")

    # # --- Step 1: Read transcript ---
    # transcript_text = Path(transcript_path).read_text(encoding="utf-8")
    # stats["raw_chars"] = len(transcript_text)
    # print(f"[ingest] Read transcript: {stats['raw_chars']} chars")

    # # --- Step 2: Clean text ---
    # clean_text = clean_transcript(transcript_text)
    # stats["clean_chars"] = len(clean_text)
    # print(f"[ingest] Cleaned: {stats['clean_chars']} chars ({stats['raw_chars'] - stats['clean_chars']} removed)")

    # # --- Step 3: Chunk ---
    # chunks = await chunk_transcript(
    #     text=clean_text,
    #     episode_id=episode_id,
    #     llm_client=openai_client,
    #     model=settings.model_segmentation,
    #     max_chunk_tokens=settings.max_chunk_tokens,
    #     min_chunk_tokens=settings.min_chunk_tokens,
    #     overlap_sentences=settings.overlap_sentences,
    #     window_size=settings.segmentation_window_size,
    #     window_overlap=settings.segmentation_window_overlap,
    # )
    # stats["chunks"] = len(chunks)
    # stats["segments"] = len(set(c.segment_index for c in chunks))
    # print(f"[ingest] Chunked: {stats['segments']} segments, {stats['chunks']} chunks")

    # # --- Step 4: Extract ---
    # print(f"[ingest] Extracting knowledge ({stats['chunks']} chunks × 3 passes)...")
    # extraction_results = await process_episode(
    #     client=openai_client,
    #     chunks=chunks,
    #     episode_title=title,
    #     episode_number=episode_number,
    #     episode_context=episode_context,
    # )

    # # Count extractions
    # total_entities = sum(len(r.entities) for r in extraction_results["entities"].values())
    # total_rels = sum(len(r.relationships) for r in extraction_results["relationships"].values())
    # total_claims = sum(len(r.claims) for r in extraction_results["claims"].values())
    # stats["entities_extracted"] = total_entities
    # stats["relationships_extracted"] = total_rels
    # stats["claims_extracted"] = total_claims
    # print(f"[ingest] Extracted: {total_entities} entities, {total_rels} relationships, {total_claims} claims")

    # # --- Step 5: Entity resolution ---
    # candidates = collect_entity_candidates(extraction_results)
    # merge_map, decisions = await resolve_entities(candidates, client=openai_client)
    # # Lets print the merge map and decisions in a pretty and readable way in the terminal for debugging and transparency
    # print("Merge Map:")
    # for src, dst in merge_map.items():
    #     print(f"  {src} -> {dst}")
    # print("Decisions:")
    # for decision in decisions:
    #     print(f"  {decision.entity_a} vs {decision.entity_b}")
    #     print(f"    Should merge: {decision.should_merge}")
    #     print(f"    Canonical name: {decision.canonical_name}")
    #     print(f"    Method: {decision.method}")
    #     print(f"    Score: {decision.score}")
    #     print(f"    Reasoning: {decision.reasoning}")

    
  

    # stats["entities_merged"] = sum(1 for d in decisions if d.should_merge)
    # stats["unique_entities"] = len(candidates) - stats["entities_merged"]
    # print(f"[ingest] Resolved: {stats['entities_merged']} merges, {stats['unique_entities']} unique entities")

    # --- Step 6: Upsert into Neo4j ---
    print("[ingest] Upserting into Neo4j...")
    async with GraphClient() as graph_client:
        # Ensure schema exists (idempotent)
        await graph_client.setup_schema()

        if run_communities:
            print("[ingest] Running community detection...")
            from src.graph.communities import detect_and_summarize_communities
            community_stats = await detect_and_summarize_communities(
                client=graph_client,
                llm_client=openai_client,
                use_seed=False,  # Incremental (CHANGE USE SEED TO FALSE IN EPISODE 48)
                gamma=1.0,  # Adjust gamma as needed
            )
            stats["communities"] = community_stats
            print(f"[ingest] Communities: {community_stats['communities_created']} created")

        # # Build segments from chunks (reconstruct for upsert)
        # segments = _reconstruct_segments(chunks)

        # # Upsert all data
        # upsert_counts = await upsert_episode_data(
        #     client=graph_client,
        #     episode_id=episode_id,
        #     episode_title=title,
        #     episode_number=episode_number,
        #     segments=segments,
        #     chunks=chunks,
        #     extraction_results=extraction_results,
        #     merge_map=merge_map,
        # )
        # stats["upserted"] = upsert_counts
        # print(f"[ingest] Upserted: {upsert_counts}")

        # # --- Step 7: Generate embeddings ---
        # print("[ingest] Generating embeddings...")
        # embed_count = await generate_embeddings(
        #     openai_client=openai_client,
        #     graph_client=graph_client,
        #     chunks=chunks,
        #     extraction_results=extraction_results,
        #     merge_map=merge_map,
        # )
        # stats["embeddings"] = embed_count
        # print(f"[ingest] Embeddings: {embed_count} generated")

        # # --- Step 8: Community detection (optional) ---
        # if run_communities:
        #     print("[ingest] Running community detection...")
        #     from src.graph.communities import detect_and_summarize_communities
        #     community_stats = await detect_and_summarize_communities(
        #         client=graph_client,
        #         llm_client=openai_client,
        #         use_seed=False,  # Incremental (CHANGE USE SEED TO FALSE IN EPISODE 48)
        #     )
        #     stats["communities"] = community_stats
        #     print(f"[ingest] Communities: {community_stats['communities_created']} created")

    print(f"[ingest] ✓ Done! Episode {episode_id} ingested successfully.")
    return stats


# =============================================================================
# HELPER: Reconstruct Segments from Chunks
# =============================================================================


def _reconstruct_segments(chunks: list) -> list[Segment]:
    """
    Reconstruct Segment objects from chunks (for upsert).
    
    The chunker produces both segments and chunks, but process_episode()
    only receives chunks. We reconstruct minimal Segment objects from
    the chunk metadata for the upsert step.
    """
    segments_map: dict[int, list] = {}
    for chunk in chunks:
        segments_map.setdefault(chunk.segment_index, []).append(chunk)

    segments = []
    for seg_idx in sorted(segments_map.keys()):
        seg_chunks = segments_map[seg_idx]
        # Combine all chunk texts for the segment
        all_sentences = []
        for c in seg_chunks:
            all_sentences.append(c.text)

        segments.append(Segment(
            segment_index=seg_idx,
            sentences=all_sentences,
            topic=f"Segment {seg_idx}",  # Topic was determined during chunking
            start_char=seg_chunks[0].start_char,
            end_char=seg_chunks[-1].end_char,
        ))

    return segments


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Command-line interface for episode ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest a single episode into the knowledge graph."
    )
    parser.add_argument(
        "--transcript", required=True,
        help="Path to the transcript text file"
    )
    parser.add_argument(
        "--episode-id", required=True,
        help="Unique episode identifier (e.g., 'ep001')"
    )
    parser.add_argument(
        "--title", required=True,
        help="Episode title"
    )
    parser.add_argument(
        "--number", type=int, required=True,
        help="Episode number"
    )
    parser.add_argument(
        "--context", default="",
        help="Known subjects/focus for this episode (helps extraction)"
    )
    parser.add_argument(
        "--communities", action="store_true",
        help="Re-run community detection after ingestion"
    )

    args = parser.parse_args()

    result = asyncio.run(ingest_episode(
        transcript_path=args.transcript,
        episode_id=args.episode_id,
        title=args.title,
        episode_number=args.number,
        episode_context=args.context,
        run_communities=args.communities,
    ))

    print("\n[ingest] Final statistics:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
