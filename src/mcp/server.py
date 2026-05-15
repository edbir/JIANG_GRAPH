"""
MCP Server — Exposes the Knowledge Graph to Client-Side LLMs.
==============================================================

PURPOSE:
    This is the INTERFACE between our knowledge graph and any LLM that
    wants to query it. It exposes 6 tools via the Model Context Protocol
    (MCP), allowing Claude, GPT, or any MCP-compatible client to search,
    compare, and analyze the graph.

WHAT IS MCP:
    Model Context Protocol is an open standard (announced by Anthropic,
    November 2024, now under Linux Foundation) for connecting AI systems
    to external data sources. It uses JSON-RPC 2.0 and defines three
    primitives: Tools, Resources, and Prompts.
    
    Our server exposes TOOLS — functions the LLM can call to retrieve
    information from the knowledge graph.

THE 6 TOOLS:
    1. search_lectures    — Semantic search across all lecture content
    2. find_claims        — Find claims by topic, type, or entity
    3. compare_entities   — Compare what Jiang says about different entities
    4. topic_timeline     — Chronological view of claims about a topic
    5. run_cypher         — Natural language → Cypher for ad-hoc queries
    6. get_schema         — Return the graph schema (so the LLM knows what's queryable)

DESIGN PRINCIPLES:
    - Domain tools for common queries (search, find_claims, compare, timeline)
      are optimized, fast, and easy for the agent to use correctly
    - Generic tools for exploration (run_cypher, get_schema) provide
      flexibility for unanticipated queries
    - Every tool caps results to prevent context overflow
    - Tools return structured data; the CLIENT-SIDE LLM handles generation

HOW TO RUN:
    python -m src.mcp.server
    
    This starts the MCP server on stdio (standard for local MCP servers).
    Configure your MCP client to connect to this server.

REFERENCE:
    MCP spec v2025-11-25. By late 2025: 97M monthly SDK downloads, 10,000+ servers.
    Neo4j provides official MCP servers (neo4j-contrib/mcp-neo4j).
    We build a CUSTOM server because our retrieval logic is domain-specific.
"""

import asyncio
import json

from mcp import FastMCP

from openai import AsyncOpenAI

from src.config import settings
from src.graph.neo4j_client import GraphClient
from src.retrieval.vector_search import (
    vector_search_with_fanout,
    search_claims,
    search_communities,
)
from src.retrieval.text2cypher import text_to_cypher, GRAPH_SCHEMA
from src.retrieval.assembler import retrieve


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

mcp = FastMCP("jiang-graphrag")

# These clients are initialized when the server starts (see main() at bottom)
_openai_client: AsyncOpenAI | None = None
_graph_client: GraphClient | None = None


async def _ensure_clients():
    """Ensure clients are initialized. Called by each tool."""
    global _openai_client, _graph_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    if _graph_client is None:
        _graph_client = GraphClient()
        await _graph_client.connect()


# =============================================================================
# TOOL 1: search_lectures
# =============================================================================


@mcp.tool()
async def search_lectures(query: str, top_k: int = 5) -> str:
    """
    Semantic search across the Predictive History lecture knowledge graph.
    
    Returns relevant passages with episode context, mentioned entities,
    and extracted claims. Use this for general questions about what
    Prof. Jiang said about any topic.
    
    Args:
        query: Natural language question or search terms
        top_k: Number of results to return (default 5, max 10)
    """
    await _ensure_clients()
    top_k = min(top_k, 10)

    passages = await vector_search_with_fanout(
        _graph_client, _openai_client, query, top_k=top_k,
    )

    # Format results for the client LLM
    results = []
    for p in passages:
        result = {
            "text": p.text,
            "score": round(p.score, 3),
            "episode": p.episode_title,
            "date": p.episode_date,
            "segment_topic": p.segment_topic,
            "entities": p.entities,
            "claims": p.claims[:3],  # Cap claims per passage
        }
        results.append(result)

    return json.dumps(results, ensure_ascii=False, indent=2)


# =============================================================================
# TOOL 2: find_claims
# =============================================================================


@mcp.tool()
async def find_claims(
    query: str,
    claim_type: str = None,
    entity: str = None,
    top_k: int = 10,
) -> str:
    """
    Find specific claims, predictions, or interpretations made by Prof. Jiang.
    
    Can filter by claim type and/or entity. Use this when you want to find
    what Jiang argued, predicted, or analyzed about a specific topic.
    
    Args:
        query: What kind of claims to search for
        claim_type: Optional filter — one of: factual_assertion, historical_interpretation, geopolitical_prediction, causal_analysis
        entity: Optional filter — only claims about this entity (e.g., "Roman Empire")
        top_k: Number of results (default 10, max 20)
    """
    await _ensure_clients()
    top_k = min(top_k, 20)

    results = await search_claims(
        _graph_client, _openai_client, query,
        top_k=top_k, claim_type=claim_type, entity_filter=entity,
    )

    return json.dumps(results, ensure_ascii=False, indent=2)


# =============================================================================
# TOOL 3: compare_entities
# =============================================================================


@mcp.tool()
async def compare_entities(topic: str, entities: list[str]) -> str:
    """
    Compare what Prof. Jiang says about different entities on a given topic.
    
    Retrieves claims from each entity side by side, enabling comparison
    of how Jiang analyzes different historical figures, empires, or events.
    
    Args:
        topic: The topic or aspect to compare on (e.g., "military strategy", "governance")
        entities: List of entity names to compare (e.g., ["Alexander the Great", "Julius Caesar"])
    """
    await _ensure_clients()

    comparison = {}
    for entity_name in entities[:4]:  # Max 4 entities
        entity_claims = await search_claims(
            _graph_client, _openai_client,
            f"{entity_name} {topic}",
            top_k=5, entity_filter=entity_name,
        )
        comparison[entity_name] = entity_claims

    return json.dumps(comparison, ensure_ascii=False, indent=2)


# =============================================================================
# TOOL 4: topic_timeline
# =============================================================================


@mcp.tool()
async def topic_timeline(
    topic: str,
    start_date: str = None,
    end_date: str = None,
) -> str:
    """
    Chronological view of how claims about a topic evolved over time.
    
    Shows claims ordered by episode date, revealing how Prof. Jiang's
    analysis or predictions about a topic developed across episodes.
    
    Args:
        topic: The topic to track over time (e.g., "United States", "democracy")
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    """
    await _ensure_clients()

    # Build a Cypher query for chronological claims
    where_clauses = ["toLower(t.name) CONTAINS toLower($topic)"]
    if start_date:
        where_clauses.append("ep.publishDate >= date($startDate)")
    if end_date:
        where_clauses.append("ep.publishDate <= date($endDate)")

    where_str = " AND ".join(where_clauses)

    cypher = f"""
        MATCH (c:Claim)-[:ABOUT]->(t)
        WHERE (t:Topic OR t:Person OR t:Organization OR t:Event)
            AND {where_str}
        MATCH (c)-[:STATED_DURING]->(ep:Episode)
        OPTIONAL MATCH (c)-[:REVISES_PREDICTION]->(revised:Claim)
        RETURN c.text AS claim,
               c.claimType AS type,
               c.stance AS stance,
               c.status AS status,
               ep.title AS episode,
               ep.publishDate AS date,
               revised.text AS revisedBy
        ORDER BY ep.publishDate ASC
        LIMIT 30
    """

    params = {"topic": topic}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    results = await _graph_client.run(cypher, params)
    return json.dumps(results, ensure_ascii=False, indent=2)


# =============================================================================
# TOOL 5: run_cypher
# =============================================================================


@mcp.tool()
async def run_cypher(question: str) -> str:
    """
    Translate a natural language question into a Cypher query and execute it.
    
    Use for analytical questions about counts, rankings, aggregations,
    and structural queries that don't fit the other tools. The system
    will generate a read-only Cypher query, execute it, and return results.
    
    Args:
        question: Natural language question (e.g., "How many episodes mention Alexander?")
    """
    await _ensure_clients()

    result = await text_to_cypher(_graph_client, _openai_client, question)

    return json.dumps(result, ensure_ascii=False, indent=2)


# =============================================================================
# TOOL 6: get_schema
# =============================================================================


@mcp.tool()
async def get_schema() -> str:
    """
    Return the knowledge graph schema — node types, relationship types,
    and their properties.
    
    Use this to understand what's in the graph before formulating queries.
    Helpful for planning which tool to use and what filters are available.
    """
    return GRAPH_SCHEMA


# =============================================================================
# SERVER STARTUP
# =============================================================================


def main():
    """
    Start the MCP server.
    
    Runs on stdio (standard for local MCP servers).
    The client (Claude Desktop, Kiro, etc.) connects via stdin/stdout.
    """
    mcp.run()


if __name__ == "__main__":
    main()
