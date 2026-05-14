"""
Multi-Pass Extraction Orchestrator.
====================================

PURPOSE:
    This is the CORE of the extraction pipeline. It takes chunks (from the chunker)
    and runs them through the 5 extraction passes, calling the OpenAI API with
    the appropriate prompts and schemas for each pass.

HOW IT CONNECTS TO THE REST:
    chunker.py produces → list[Chunk]
    extractor.py consumes those chunks and produces → extracted knowledge
    
    For each chunk, it runs:
        Pass 1: Entity Extraction      (what entities are mentioned?)
        Pass 2: Relationship Extraction (how are they connected?)
        Pass 3: Claim Extraction        (what does Jiang argue/predict?)
    
    After ALL chunks in an episode are processed:
        Pass 4: Coreference Resolution  (resolve "he", "the empire", etc.)
    
    After ALL episodes:
        Pass 5: Entity Resolution       (deduplicate entities across episodes)

WHY MULTI-PASS (not single-pass):
    Research shows multi-pass is strictly superior:
    - ECR: +16% improvement in triple F1 over single-pass
    - KGGen: 18% improvement over GraphRAG via multi-stage extraction
    - SAC-KG: 89.32% precision with Generator→Verifier→Pruner pipeline
    
    Each pass has a FOCUSED task. The LLM does better when asked to do
    one thing well rather than everything at once.

ARCHITECTURE:
    - Each pass is an independent async function
    - All passes use OpenAI Structured Outputs (guaranteed JSON compliance)
    - Results are cached by content hash (reprocessing is free)
    - The orchestrator handles the flow between passes

⚠️  THIS MODULE REQUIRES AN OPENAI API KEY TO RUN.
    Without a key, you can still:
    - Read the code to understand the flow
    - Test with mock responses
    - Use the individual pass functions with fake clients
"""

import json
import hashlib
from pathlib import Path

from src.preprocessing.chunker import Chunk
from src.extraction.schemas import (
    EntityExtractionResult,
    RelationshipExtractionResult,
    ClaimExtractionResult,
    CoreferenceResult,
    EntityResolutionBatch,
)
from src.extraction.prompts import (
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_USER,
    RELATIONSHIP_EXTRACTION_SYSTEM,
    RELATIONSHIP_EXTRACTION_USER,
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    COREFERENCE_SYSTEM,
    COREFERENCE_USER,
    ENTITY_RESOLUTION_SYSTEM,
    ENTITY_RESOLUTION_USER,
)
from src.config import settings


# =============================================================================
# RESPONSE CACHING
# =============================================================================
# Every LLM call is cached by hash(prompt + parameters).
# This means:
# - Reprocessing an episode costs $0 (cache hit)
# - Debugging is easy (inspect cached responses)
# - The pipeline is idempotent (same input → same output)
#
# Cache is stored as JSON files in data/cache/
# =============================================================================

CACHE_DIR = Path("data/cache")


def _cache_key(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Generate a unique cache key for an LLM call.
    
    We hash the full prompt content + model name. If any of these change,
    we get a different key (cache miss) and re-call the API.
    """
    content = f"{model}::{system_prompt}::{user_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def _get_cached(cache_key: str) -> dict | None:
    """
    Check if we have a cached response for this call.
    Returns the parsed JSON response, or None if not cached.
    """
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def _save_cache(cache_key: str, response: dict) -> None:
    """Save an LLM response to the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps(response, ensure_ascii=False, indent=2))


# =============================================================================
# GENERIC LLM CALL (used by all passes)
# =============================================================================


async def _call_llm(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: type,
    schema_name: str,
) -> dict:
    """
    Make a single LLM call with structured output and caching.
    
    This is the ONLY function that actually talks to the OpenAI API.
    All 5 passes use this same function — they just pass different
    prompts and schemas.
    
    FLOW:
    1. Check cache → if hit, return cached response (no API call)
    2. Call OpenAI API with structured output format
    3. Parse the response JSON
    4. Save to cache
    5. Return parsed result
    
    Parameters:
        client:          OpenAI AsyncClient
        model:           Which model to use (e.g., "gpt-4o-mini")
        system_prompt:   The system message (static instructions)
        user_prompt:     The user message (chunk text + context)
        response_schema: The Pydantic model class for structured output
        schema_name:     Name for the JSON schema (e.g., "EntityExtractionResult")
    
    Returns:
        Parsed JSON response as a dict
    """
    # Step 1: Check cache
    key = _cache_key(system_prompt, user_prompt, model)
    cached = _get_cached(key)
    if cached is not None:
        return cached

    # Step 2: Call the API
    response = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_schema.model_json_schema(),
            }
        },
    )

    # Step 3: Parse the response
    result = json.loads(response.choices[0].message.content)

    # Step 4: Cache it
    _save_cache(key, result)

    # Step 5: Return
    return result


# =============================================================================
# PASS 1 — ENTITY EXTRACTION
# =============================================================================


async def extract_entities(
    client,
    chunk: Chunk,
    episode_title: str = "",
    episode_number: int = 0,
    episode_context: str = "",
) -> EntityExtractionResult:
    """
    Pass 1: Extract entities from a single chunk.
    
    WHAT IT DOES:
        Sends the chunk text to GPT-4o-mini and asks:
        "What persons, organizations, topics, and events are mentioned here?"
    
    WHAT IT RETURNS:
        An EntityExtractionResult containing a list of Entity objects,
        each with name, type, description, aliases, and role/category.
    
    Parameters:
        client:          OpenAI AsyncClient
        chunk:           A Chunk object from the chunker
        episode_title:   Title of the episode (gives LLM context)
        episode_number:  Episode number
        episode_context: Known subjects/focus for this episode
    
    Returns:
        EntityExtractionResult (validated Pydantic model)
    """
    # Format the user prompt with chunk text and episode context
    user_prompt = ENTITY_EXTRACTION_USER.format(
        episode_title=episode_title,
        episode_number=episode_number,
        episode_context=episode_context,
        chunk_text=chunk.text,
    )

    # Call the LLM
    result = await _call_llm(
        client=client,
        model=settings.model_cheap,  # gpt-4o-mini (high volume, cheap)
        system_prompt=ENTITY_EXTRACTION_SYSTEM,
        user_prompt=user_prompt,
        response_schema=EntityExtractionResult,
        schema_name="EntityExtractionResult",
    )

    # Validate and return as Pydantic model
    return EntityExtractionResult(**result)


# =============================================================================
# PASS 2 — RELATIONSHIP EXTRACTION
# =============================================================================


async def extract_relationships(
    client,
    chunk: Chunk,
    entities: EntityExtractionResult,
) -> RelationshipExtractionResult:
    """
    Pass 2: Extract relationships between entities in a chunk.
    
    WHAT IT DOES:
        Takes the SAME chunk as Pass 1, PLUS the entities already found,
        and asks: "What relationships exist between these entities?"
    
    WHY ENTITIES AS CONTEXT:
        By giving the LLM the entity list from Pass 1, it focuses on
        finding relationships between KNOWN entities rather than
        re-extracting entities or inventing new ones.
    
    Parameters:
        client:    OpenAI AsyncClient
        chunk:     The same Chunk used in Pass 1
        entities:  The EntityExtractionResult from Pass 1
    
    Returns:
        RelationshipExtractionResult (validated Pydantic model)
    """
    # Format entities as context for the prompt
    # We give the LLM a simple list: "- Alexander the Great (person)"
    entities_context = "\n".join(
        f"- {e.name} ({e.entity_type.value})"
        for e in entities.entities
    )

    user_prompt = RELATIONSHIP_EXTRACTION_USER.format(
        entities_context=entities_context,
        chunk_text=chunk.text,
    )

    result = await _call_llm(
        client=client,
        model=settings.model_cheap,  # gpt-4o-mini
        system_prompt=RELATIONSHIP_EXTRACTION_SYSTEM,
        user_prompt=user_prompt,
        response_schema=RelationshipExtractionResult,
        schema_name="RelationshipExtractionResult",
    )

    return RelationshipExtractionResult(**result)


# =============================================================================
# PASS 3 — CLAIM/OPINION EXTRACTION
# =============================================================================


async def extract_claims(
    client,
    chunk: Chunk,
    entities: EntityExtractionResult,
) -> ClaimExtractionResult:
    """
    Pass 3: Extract claims, interpretations, and predictions.
    
    WHAT IT DOES:
        This is the MOST IMPORTANT pass. It identifies what Prof. Jiang
        ARGUES, INTERPRETS, PREDICTS, and ANALYZES — not just what
        entities he mentions.
    
    WHY GPT-4o (not mini):
        Claim extraction requires nuanced understanding:
        - Distinguishing fact from opinion
        - Detecting stance (critical vs. favorable vs. speculative)
        - Identifying implicit predictions
        - Understanding causal reasoning
        
        GPT-4o-mini struggles with these subtleties. GPT-4o handles them well.
    
    Parameters:
        client:    OpenAI AsyncClient
        chunk:     The same Chunk used in Pass 1
        entities:  The EntityExtractionResult from Pass 1 (for reference)
    
    Returns:
        ClaimExtractionResult (validated Pydantic model)
    """
    entities_context = "\n".join(
        f"- {e.name} ({e.entity_type.value})"
        for e in entities.entities
    )

    user_prompt = CLAIM_EXTRACTION_USER.format(
        entities_context=entities_context,
        chunk_text=chunk.text,
    )

    result = await _call_llm(
        client=client,
        model=settings.model_strong,  # gpt-4o (nuanced reasoning needed)
        system_prompt=CLAIM_EXTRACTION_SYSTEM,
        user_prompt=user_prompt,
        response_schema=ClaimExtractionResult,
        schema_name="ClaimExtractionResult",
    )

    return ClaimExtractionResult(**result)


# =============================================================================
# PASS 4 — COREFERENCE RESOLUTION
# =============================================================================


async def resolve_coreferences(
    client,
    chunks_window: list[Chunk],
    entities_per_chunk: dict[str, EntityExtractionResult],
) -> CoreferenceResult:
    """
    Pass 4: Resolve pronouns and indirect references to canonical entities.
    
    WHAT IT DOES:
        Takes a WINDOW of consecutive chunks (typically 3-5) and resolves
        ambiguous references like "he", "the empire", "that city" to their
        canonical entity names.
    
    WHY A WINDOW (not single chunk):
        Pronouns often refer to entities mentioned in PREVIOUS chunks.
        "Alexander conquered Persia." [chunk N]
        "He then marched toward India." [chunk N+1]
        
        Without the window, we can't resolve "He" in chunk N+1.
    
    WHY AFTER PASSES 1-3 (not before):
        We need the entity list from Pass 1 to know WHAT entities exist.
        The LLM resolves references TO those known entities.
    
    Parameters:
        client:            OpenAI AsyncClient
        chunks_window:     A sliding window of 3-5 consecutive Chunks
        entities_per_chunk: Dict mapping chunk_id → EntityExtractionResult
    
    Returns:
        CoreferenceResult (validated Pydantic model)
    """
    # Collect all known entities across the window
    all_entities: list[str] = []
    for chunk in chunks_window:
        if chunk.chunk_id in entities_per_chunk:
            for entity in entities_per_chunk[chunk.chunk_id].entities:
                if entity.name not in all_entities:
                    all_entities.append(entity.name)

    entities_context = "\n".join(f"- {name}" for name in all_entities)

    # Format chunks as numbered text blocks
    chunks_text = "\n\n".join(
        f"[Chunk {i}]:\n{chunk.text}"
        for i, chunk in enumerate(chunks_window)
    )

    user_prompt = COREFERENCE_USER.format(
        entities_context=entities_context,
        chunks_text=chunks_text,
    )

    result = await _call_llm(
        client=client,
        model=settings.model_strong,  # gpt-4o (deep contextual reasoning)
        system_prompt=COREFERENCE_SYSTEM,
        user_prompt=user_prompt,
        response_schema=CoreferenceResult,
        schema_name="CoreferenceResult",
    )

    return CoreferenceResult(**result)


# =============================================================================
# PASS 5 — ENTITY RESOLUTION (LLM stage)
# =============================================================================


async def resolve_entity_pair(
    client,
    entity_a_name: str,
    entity_a_type: str,
    entity_a_description: str,
    entity_a_aliases: list[str],
    entity_b_name: str,
    entity_b_type: str,
    entity_b_description: str,
    entity_b_aliases: list[str],
) -> EntityResolutionBatch:
    """
    Pass 5 (LLM stage): Decide if two entities should be merged.
    
    WHEN THIS IS CALLED:
        Only for the ~5% of entity pairs that the cheaper methods
        (exact match, fuzzy string, embedding similarity) could NOT resolve.
        These are genuinely ambiguous cases that need LLM judgment.
    
    EXAMPLE:
        Entity A: "Philip II" (person) — "King of Macedon, father of Alexander"
        Entity B: "Philip of Macedon" (person) — "Macedonian ruler who unified Greece"
        → should_merge: True, canonical: "Philip II of Macedon"
    
    Parameters:
        client:              OpenAI AsyncClient
        entity_a_*:          Properties of the first entity
        entity_b_*:          Properties of the second entity
    
    Returns:
        EntityResolutionBatch with the merge decision
    """
    user_prompt = ENTITY_RESOLUTION_USER.format(
        entity_a_name=entity_a_name,
        entity_a_type=entity_a_type,
        entity_a_description=entity_a_description,
        entity_a_aliases=", ".join(entity_a_aliases) if entity_a_aliases else "none",
        entity_b_name=entity_b_name,
        entity_b_type=entity_b_type,
        entity_b_description=entity_b_description,
        entity_b_aliases=", ".join(entity_b_aliases) if entity_b_aliases else "none",
    )

    result = await _call_llm(
        client=client,
        model=settings.model_cheap,  # gpt-4o-mini (binary decision, not nuanced)
        system_prompt=ENTITY_RESOLUTION_SYSTEM,
        user_prompt=user_prompt,
        response_schema=EntityResolutionBatch,
        schema_name="EntityResolutionBatch",
    )

    return EntityResolutionBatch(**result)


# =============================================================================
# EPISODE-LEVEL ORCHESTRATION
# =============================================================================
# This ties all passes together for processing a complete episode.
# It's the function you call from scripts/ingest_episode.py
# =============================================================================


async def process_episode(
    client,
    chunks: list[Chunk],
    episode_title: str = "",
    episode_number: int = 0,
    episode_context: str = "",
) -> dict:
    """
    Process all chunks of an episode through the full extraction pipeline.
    
    FLOW:
    1. For each chunk: run Pass 1 (entities), Pass 2 (relationships), Pass 3 (claims)
    2. After all chunks: run Pass 4 (coreference) on sliding windows
    3. Return all extracted knowledge organized by chunk
    
    Pass 5 (entity resolution) is NOT run here — it happens AFTER all episodes
    are processed, because it deduplicates entities ACROSS episodes.
    
    Parameters:
        client:          OpenAI AsyncClient
        chunks:          List of Chunks from the chunker (one episode)
        episode_title:   Title of the episode
        episode_number:  Episode number
        episode_context: Known subjects/focus
    
    Returns:
        Dict with all extraction results:
        {
            "entities": {chunk_id: EntityExtractionResult, ...},
            "relationships": {chunk_id: RelationshipExtractionResult, ...},
            "claims": {chunk_id: ClaimExtractionResult, ...},
            "coreferences": [CoreferenceResult, ...],
        }
    """
    # Storage for results
    all_entities: dict[str, EntityExtractionResult] = {}
    all_relationships: dict[str, RelationshipExtractionResult] = {}
    all_claims: dict[str, ClaimExtractionResult] = {}
    all_coreferences: list[CoreferenceResult] = []

    # ----- Passes 1-3: Per-chunk extraction -----
    # Process each chunk sequentially (for prompt caching benefits)
    # Sequential processing means the system prompt stays in OpenAI's cache
    # between calls, giving us ~50% discount on those tokens.

    for chunk in chunks:
        # Pass 1: What entities are in this chunk?
        entities = await extract_entities(
            client=client,
            chunk=chunk,
            episode_title=episode_title,
            episode_number=episode_number,
            episode_context=episode_context,
        )
        all_entities[chunk.chunk_id] = entities

        # Pass 2: How are those entities related?
        relationships = await extract_relationships(
            client=client,
            chunk=chunk,
            entities=entities,
        )
        all_relationships[chunk.chunk_id] = relationships

        # Pass 3: What does Jiang claim/argue/predict?
        claims = await extract_claims(
            client=client,
            chunk=chunk,
            entities=entities,
        )
        all_claims[chunk.chunk_id] = claims

    # ----- Pass 4: Coreference resolution (sliding window over chunks) -----
    # We slide a window of N chunks across the episode and resolve
    # pronouns/references in each window.

    window_size = settings.coreference_window_chunks  # default: 4

    for i in range(0, len(chunks), window_size - 1):
        # Take a window of chunks (overlapping by 1 for continuity)
        window = chunks[i:i + window_size]

        if len(window) < 2:
            # Need at least 2 chunks for coreference to be meaningful
            continue

        coref_result = await resolve_coreferences(
            client=client,
            chunks_window=window,
            entities_per_chunk=all_entities,
        )
        all_coreferences.append(coref_result)

    # ----- Return all results -----
    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "claims": all_claims,
        "coreferences": all_coreferences,
    }
