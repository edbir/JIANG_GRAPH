"""
Vector Search + Graph Fan-Out — The core retrieval pattern.
=============================================================

PURPOSE:
    Find relevant passages via semantic similarity (vector search),
    then expand into the graph neighborhood for structured context.

THE PATTERN:
    1. Embed the user's query
    2. Search the utterance_embeddings vector index (top-K similar chunks)
    3. For each matching chunk, traverse the graph to collect:
       - Episode metadata (title, date)
       - Segment topic
       - Entities mentioned in that chunk
       - Claims extracted from that chunk
       - Topics those claims reference
    
    This gives us BOTH the raw text passage AND the structured knowledge
    around it — something flat vector RAG cannot do.

WHY "FAN-OUT":
    Vector search finds the entry point (the chunk). Graph traversal
    "fans out" from that point into the surrounding knowledge structure.
    It's like finding a page in a book (vector) and then reading the
    table of contents, index, and footnotes (graph).

ALSO SUPPORTS:
    - Entity-filtered search (only chunks mentioning a specific entity)
    - Claim-level search (search claim embeddings directly)
    - Hybrid search (vector + full-text keyword, fused with RRF)
"""

from dataclasses import dataclass

from src.graph.neo4j_client import GraphClient
from src.config import settings


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================


@dataclass
class RetrievedPassage:
    """
    A single retrieved passage with its graph context.
    This is what the assembler works with to build the final context.
    """
    text: str                   # The chunk text
    score: float                # Vector similarity score (0-1)
    episode_title: str          # Which episode
    episode_date: str | None    # When published
    segment_topic: str | None   # What topic this segment covers
    chunk_id: str               # For deduplication
    claims: list[dict]          # Claims from this chunk: [{text, type, stance, topic}]
    entities: list[str]         # Entity names mentioned in this chunk


# =============================================================================
# CORE: VECTOR + GRAPH FAN-OUT
# =============================================================================


async def vector_search_with_fanout(
    graph_client: GraphClient,
    openai_client,
    query: str,
    top_k: int = 10,
    entity_filter: str = None,
) -> list[RetrievedPassage]:
    """
    The primary retrieval pattern: vector search → graph fan-out.
    
    HOW IT WORKS:
    1. Embed the query using OpenAI
    2. Query Neo4j's vector index for the top-K most similar utterances
    3. For each utterance, traverse the graph to collect context:
       - Episode (title, date)
       - Segment (topic)
       - Entities mentioned (MENTIONS relationships)
       - Claims extracted (EXTRACTED_FROM relationships)
       - Topics of those claims (ABOUT relationships)
    
    OPTIONAL ENTITY FILTER:
        If entity_filter is provided (e.g., "Alexander the Great"),
        only return chunks that MENTION that entity. This is used
        for factual queries about a specific entity.
    
    Parameters:
        graph_client:   GraphClient instance
        openai_client:  OpenAI AsyncClient (for embedding the query)
        query:          The user's question (will be embedded)
        top_k:          How many results to return
        entity_filter:  Optional entity name to filter by
    
    Returns:
        List of RetrievedPassage objects, sorted by score (descending)
    """
    # Step 1: Embed the query
    embedding_response = await openai_client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    # Step 2 + 3: Vector search with graph fan-out (single Cypher query)
    # This combines the vector search and graph traversal in one query
    # for efficiency (one round-trip to Neo4j instead of many).

    if entity_filter:
        # Filtered: only chunks mentioning a specific entity
        cypher = """
            CALL db.index.vector.queryNodes('utterance_embeddings', $topK * 3, $embedding)
            YIELD node AS utterance, score
            
            // Filter: only chunks that mention the target entity
            MATCH (utterance)-[:MENTIONS]->(entity)
            WHERE entity.name = $entityFilter OR $entityFilter IN entity.aliases
            
            // Fan-out: get episode and segment context
            OPTIONAL MATCH (seg:Segment)-[:HAS_CHUNK]->(utterance)
            OPTIONAL MATCH (ep:Episode)-[:HAS_SEGMENT]->(seg)
            
            // Fan-out: get all entities mentioned
            OPTIONAL MATCH (utterance)-[:MENTIONS]->(e)
            
            // Fan-out: get claims from this chunk
            OPTIONAL MATCH (claim:Claim)-[:EXTRACTED_FROM]->(utterance)
            OPTIONAL MATCH (claim)-[:ABOUT]->(topic)
            
            RETURN utterance.utteranceId AS chunkId,
                   utterance.text AS text,
                   score,
                   ep.title AS episodeTitle,
                   ep.publishDate AS episodeDate,
                   seg.topic AS segmentTopic,
                   collect(DISTINCT e.name) AS entities,
                   collect(DISTINCT {
                       text: claim.text,
                       type: claim.claimType,
                       stance: claim.stance,
                       topic: topic.name
                   }) AS claims
            ORDER BY score DESC
            LIMIT $topK
        """
        params = {
            "embedding": query_embedding,
            "topK": top_k,
            "entityFilter": entity_filter,
        }
    else:
        # Unfiltered: pure semantic search
        cypher = """
            CALL db.index.vector.queryNodes('utterance_embeddings', $topK, $embedding)
            YIELD node AS utterance, score
            
            // Fan-out: get episode and segment context
            OPTIONAL MATCH (seg:Segment)-[:HAS_CHUNK]->(utterance)
            OPTIONAL MATCH (ep:Episode)-[:HAS_SEGMENT]->(seg)
            
            // Fan-out: get all entities mentioned
            OPTIONAL MATCH (utterance)-[:MENTIONS]->(e)
            
            // Fan-out: get claims from this chunk
            OPTIONAL MATCH (claim:Claim)-[:EXTRACTED_FROM]->(utterance)
            OPTIONAL MATCH (claim)-[:ABOUT]->(topic)
            
            RETURN utterance.utteranceId AS chunkId,
                   utterance.text AS text,
                   score,
                   ep.title AS episodeTitle,
                   ep.publishDate AS episodeDate,
                   seg.topic AS segmentTopic,
                   collect(DISTINCT e.name) AS entities,
                   collect(DISTINCT {
                       text: claim.text,
                       type: claim.claimType,
                       stance: claim.stance,
                       topic: topic.name
                   }) AS claims
            ORDER BY score DESC
            LIMIT $topK
        """
        params = {
            "embedding": query_embedding,
            "topK": top_k,
        }

    results = await graph_client.run(cypher, params)

    # Convert to RetrievedPassage objects
    passages: list[RetrievedPassage] = []
    for record in results:
        # Filter out null claims (from OPTIONAL MATCH)
        claims = [c for c in record.get("claims", []) if c.get("text")]

        passages.append(RetrievedPassage(
            text=record["text"],
            score=record["score"],
            episode_title=record.get("episodeTitle", ""),
            episode_date=record.get("episodeDate"),
            segment_topic=record.get("segmentTopic"),
            chunk_id=record["chunkId"],
            claims=claims,
            entities=[e for e in record.get("entities", []) if e],
        ))

    return passages


# =============================================================================
# CLAIM-LEVEL SEARCH
# =============================================================================


async def search_claims(
    graph_client: GraphClient,
    openai_client,
    query: str,
    top_k: int = 10,
    claim_type: str = None,
    entity_filter: str = None,
) -> list[dict]:
    """
    Search directly on claim embeddings (not chunk embeddings).
    
    Useful for comparative and temporal queries where we want
    specific claims rather than full passages.
    
    Parameters:
        graph_client:   GraphClient instance
        openai_client:  OpenAI AsyncClient
        query:          The user's question
        top_k:          How many claims to return
        claim_type:     Optional filter: "factual_assertion", "geopolitical_prediction", etc.
        entity_filter:  Optional: only claims ABOUT a specific entity
    
    Returns:
        List of claim dicts with metadata
    """
    # Embed the query
    embedding_response = await openai_client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    # Build Cypher with optional filters
    where_clauses = []
    if claim_type:
        where_clauses.append("claim.claimType = $claimType")
    if entity_filter:
        where_clauses.append("""
            EXISTS {
                MATCH (claim)-[:ABOUT]->(e)
                WHERE e.name = $entityFilter OR $entityFilter IN e.aliases
            }
        """)

    where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    cypher = f"""
        CALL db.index.vector.queryNodes('claim_embeddings', $topK * 2, $embedding)
        YIELD node AS claim, score
        {where_str}
        
        OPTIONAL MATCH (claim)-[:EXTRACTED_FROM]->(u:Utterance)
        OPTIONAL MATCH (claim)-[:STATED_DURING]->(ep:Episode)
        OPTIONAL MATCH (claim)-[:ABOUT]->(entity)
        
        RETURN claim.claimId AS claimId,
               claim.text AS text,
               claim.claimType AS claimType,
               claim.stance AS stance,
               claim.status AS status,
               score,
               ep.title AS episodeTitle,
               ep.publishDate AS episodeDate,
               collect(DISTINCT entity.name) AS entities
        ORDER BY score DESC
        LIMIT $topK
    """

    params = {
        "embedding": query_embedding,
        "topK": top_k,
    }
    if claim_type:
        params["claimType"] = claim_type
    if entity_filter:
        params["entityFilter"] = entity_filter

    return await graph_client.run(cypher, params)


# =============================================================================
# COMMUNITY SEARCH
# =============================================================================


async def search_communities(
    graph_client: GraphClient,
    openai_client,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Search community summaries by semantic similarity.
    
    Used for thematic queries ("What are the main themes?").
    Returns community summaries ranked by relevance to the query.
    
    Parameters:
        graph_client:   GraphClient instance
        openai_client:  OpenAI AsyncClient
        query:          The user's question
        top_k:          How many communities to return
    
    Returns:
        List of community dicts with summary and member info
    """
    # Embed the query
    embedding_response = await openai_client.embeddings.create(
        model=settings.embedding_model,
        input=query,
    )
    query_embedding = embedding_response.data[0].embedding

    cypher = """
        CALL db.index.vector.queryNodes('community_embeddings', $topK, $embedding)
        YIELD node AS community, score
        
        // Get member names for context
        OPTIONAL MATCH (entity)-[:IN_COMMUNITY]->(community)
        
        RETURN community.communityId AS communityId,
               community.summary AS summary,
               community.memberCount AS memberCount,
               score,
               collect(DISTINCT entity.name)[..10] AS topMembers
        ORDER BY score DESC
        LIMIT $topK
    """

    return await graph_client.run(cypher, {
        "embedding": query_embedding,
        "topK": top_k,
    })
