"""
Context Assembler — Fuses retrieval results and builds final context.
======================================================================

PURPOSE:
    Takes results from multiple retrieval strategies (vector search,
    claim search, Text-to-Cypher, community summaries) and:
    1. Fuses them using Reciprocal Rank Fusion (RRF)
    2. Deduplicates overlapping results
    3. Assembles a structured context string ready for LLM generation

WHY RRF (Reciprocal Rank Fusion):
    When multiple retrieval strategies each return a ranked list,
    we need to combine them into a single ranking. RRF does this
    without needing score normalization (scores from different
    strategies aren't comparable).
    
    Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
    Where k=60 (standard constant) and rank_i is the rank in list i.
    
    A document ranked #1 in two lists gets: 1/61 + 1/61 = 0.033
    A document ranked #1 in one list and #5 in another: 1/61 + 1/65 = 0.032
    
    RRF naturally favors documents that appear in MULTIPLE lists
    (consensus across strategies = likely relevant).

REFERENCE:
    Original RRF paper: Cormack et al., SIGIR 2009.
    RAG-Fusion (Rackauckas, 2024): extends RRF to RAG, reports
    +8-10% accuracy and +30-40% comprehensiveness.

CONTEXT FORMAT:
    The assembled context separates different types of information:
    
    ## Relevant Passages
    [Episode, Date]: "chunk text..."
    
    ## Knowledge Graph Context
    - Topic: claim text (stance, type)
    
    ## Structured Query Results
    (from Text-to-Cypher)
    
    ## Thematic Context
    Community: summary
    
    This structured format helps the generation LLM distinguish between
    raw text evidence and structured graph knowledge.
"""

from dataclasses import dataclass

from src.retrieval.classifier import QueryType, QueryClassification
from src.retrieval.vector_search import (
    RetrievedPassage,
    vector_search_with_fanout,
    search_claims,
    search_communities,
)
from src.retrieval.text2cypher import text_to_cypher
from src.graph.neo4j_client import GraphClient


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    id_key: str = "id",
    k: int = 60,
) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.
    
    HOW IT WORKS:
        For each document d that appears in any list:
            RRF_score(d) = sum over all lists of: 1 / (k + rank_in_that_list)
        
        Documents that appear in MULTIPLE lists get higher scores
        (they're relevant according to multiple strategies).
        
        k=60 is the standard constant from the original paper.
        It dampens the effect of high ranks (rank 1 vs rank 2 matters less
        than rank 1 vs rank 50).
    
    Parameters:
        result_lists: List of ranked result lists. Each result must have an 'id' field.
        id_key:       The key to use for identifying unique results
        k:            RRF constant (default 60, from original paper)
    
    Returns:
        Single fused list, sorted by RRF score (descending)
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            item_id = item.get(id_key, str(rank))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            # Keep the item data (first occurrence wins)
            if item_id not in items:
                items[item_id] = item

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        {**items[item_id], "_rrf_score": scores[item_id]}
        for item_id in sorted_ids
    ]


# =============================================================================
# CONTEXT ASSEMBLY
# =============================================================================


def assemble_context(
    passages: list[RetrievedPassage] = None,
    claims: list[dict] = None,
    t2c_results: dict = None,
    communities: list[dict] = None,
    max_passages: int = 5,
    max_claims: int = 10,
    max_communities: int = 3,
) -> str:
    """
    Assemble the final context string from all retrieval results.
    
    The context is structured into sections so the generation LLM
    can distinguish between raw text evidence and structured knowledge.
    
    Parameters:
        passages:       Results from vector_search_with_fanout()
        claims:         Results from search_claims()
        t2c_results:    Results from text_to_cypher()
        communities:    Results from search_communities()
        max_passages:   How many passages to include
        max_claims:     How many claims to include
        max_communities: How many community summaries to include
    
    Returns:
        Formatted context string ready for LLM generation
    """
    parts: list[str] = []

    # --- Section 1: Relevant Passages ---
    if passages:
        parts.append("## Relevant Passages")
        for p in passages[:max_passages]:
            header = f"[{p.episode_title}"
            if p.episode_date:
                header += f", {p.episode_date}"
            header += "]"
            parts.append(f"{header}: {p.text}")

            # Include claims from this passage if any
            if p.claims:
                for claim in p.claims[:3]:  # Max 3 claims per passage
                    if claim.get("text"):
                        topic = claim.get("topic", "")
                        stance = claim.get("stance", "")
                        parts.append(f"  → Claim ({stance}): {claim['text']}")
        parts.append("")

    # --- Section 2: Knowledge Graph Context (claims) ---
    if claims:
        parts.append("## Knowledge Graph Context")
        for claim in claims[:max_claims]:
            entities_str = ", ".join(claim.get("entities", []))
            parts.append(
                f"- [{claim.get('claimType', '')}] {claim.get('text', '')} "
                f"(stance: {claim.get('stance', '')}, entities: {entities_str})"
            )
        parts.append("")

    # --- Section 3: Structured Query Results ---
    if t2c_results and t2c_results.get("success") and t2c_results.get("results"):
        parts.append("## Structured Query Results")
        parts.append(f"Query: {t2c_results.get('query', '')}")
        for row in t2c_results["results"][:10]:
            parts.append(f"  {row}")
        parts.append("")

    # --- Section 4: Thematic Context ---
    if communities:
        parts.append("## Thematic Context")
        for comm in communities[:max_communities]:
            members = ", ".join(comm.get("topMembers", [])[:5])
            parts.append(
                f"- **{comm.get('summary', 'No summary')}** "
                f"(members: {members})"
            )
        parts.append("")

    return "\n".join(parts) if parts else "No relevant context found."


# =============================================================================
# FULL RETRIEVAL PIPELINE
# =============================================================================


async def retrieve(
    graph_client: GraphClient,
    openai_client,
    query: str,
    classification: QueryClassification = None,
) -> str:
    """
    Full retrieval pipeline: classify → route → retrieve → fuse → assemble.
    
    This is the MAIN ENTRY POINT for the retrieval layer.
    It takes a user query and returns assembled context ready for generation.
    
    FLOW:
    1. Classify the query (if not already classified)
    2. Based on type, activate appropriate strategies
    3. Run all active strategies
    4. Assemble context from results
    
    STRATEGY ROUTING:
        Factual     → vector search (+ entity filter if entity mentioned)
        Comparative → claim search (filtered by mentioned entities)
        Temporal    → claim search (sorted chronologically)
        Thematic    → community search
        
        Text-to-Cypher is activated for comparative and temporal queries
        as a SUPPLEMENT (not replacement) to the primary strategy.
    
    Parameters:
        graph_client:    GraphClient instance
        openai_client:   OpenAI AsyncClient
        query:           The user's natural language question
        classification:  Pre-computed classification (optional, will classify if None)
    
    Returns:
        Assembled context string ready for LLM generation
    """
    # Step 1: Classify (if not already done)
    if classification is None:
        from src.retrieval.classifier import classify_query
        classification = await classify_query(openai_client, query)

    # Step 2: Route to strategies based on query type
    passages: list[RetrievedPassage] = []
    claims: list[dict] = []
    t2c_results: dict = {}
    communities: list[dict] = []

    # --- Factual queries: vector search with optional entity filter ---
    if classification.query_type == QueryType.FACTUAL:
        entity_filter = classification.mentioned_entities[0] if classification.mentioned_entities else None
        passages = await vector_search_with_fanout(
            graph_client, openai_client, query,
            top_k=10, entity_filter=entity_filter,
        )

    # --- Comparative queries: claim search + Text-to-Cypher ---
    elif classification.query_type == QueryType.COMPARATIVE:
        # Search claims for each mentioned entity
        for entity in classification.mentioned_entities[:3]:
            entity_claims = await search_claims(
                graph_client, openai_client, query,
                top_k=5, entity_filter=entity,
            )
            claims.extend(entity_claims)

        # Also try Text-to-Cypher for structured comparison
        t2c_results = await text_to_cypher(graph_client, openai_client, query)

        # Also get passages for raw text context
        passages = await vector_search_with_fanout(
            graph_client, openai_client, query, top_k=5,
        )

    # --- Temporal queries: chronological claim retrieval ---
    elif classification.query_type == QueryType.TEMPORAL:
        # Search claims about the topic, will be sorted by date
        topic = classification.topics[0] if classification.topics else query
        claims_results = await search_claims(
            graph_client, openai_client, topic, top_k=15,
        )
        claims = claims_results

        # Text-to-Cypher for structured temporal query
        t2c_results = await text_to_cypher(graph_client, openai_client, query)

    # --- Thematic queries: community summaries ---
    elif classification.query_type == QueryType.THEMATIC:
        communities = await search_communities(
            graph_client, openai_client, query, top_k=5,
        )

        # Also get some passages for grounding
        passages = await vector_search_with_fanout(
            graph_client, openai_client, query, top_k=3,
        )

    # Step 3: Assemble context
    context = assemble_context(
        passages=passages,
        claims=claims,
        t2c_results=t2c_results,
        communities=communities,
    )

    return context
