"""
Text-to-Cypher — Natural Language to Graph Queries with Self-Healing.
======================================================================

PURPOSE:
    Translates natural language questions into Cypher queries, executes them,
    and returns structured results from the graph.

WHEN USED:
    For analytical queries that map naturally to graph patterns:
    - "How many episodes discuss the Roman Empire?"
    - "Which topics have the most claims?"
    - "What predictions were verified as correct?"
    - "Show me all causal chains involving Alexander"

    These are queries where the STRUCTURE of the graph is the answer,
    not a text passage.

SELF-HEALING:
    Generated Cypher may fail (syntax errors, non-existent properties, etc.).
    When it does:
    1. Capture the error message
    2. Feed error + original query + schema back to the LLM
    3. LLM generates a corrected query
    4. Retry (up to 2 times)
    
    This "self-healing" pattern is standard practice in Neo4j's Text2Cypher
    documentation and dramatically improves reliability.

SAFETY:
    - Only READ queries are allowed (no CREATE, MERGE, DELETE, SET, REMOVE)
    - Results are capped at 50 rows to prevent context overflow
    - The schema is always included so the LLM knows what's queryable

REFERENCE:
    Neo4j Text2Cypher 2024 dataset (Ozsoy et al.): 44,387 instances.
    GPT-4o achieved best performance; described as "the most flexible,
    but also the most unreliable" retrieval pattern.
"""

import json
import re

from src.graph.neo4j_client import GraphClient
from src.config import settings


# =============================================================================
# GRAPH SCHEMA (for the LLM to know what's queryable)
# =============================================================================
# This is a text representation of our graph schema that we include in every
# Text-to-Cypher prompt. The LLM needs to know what node types, relationship
# types, and properties exist to generate valid Cypher.
# =============================================================================

GRAPH_SCHEMA = """
NODE TYPES:
- (:Episode {episodeId, title, publishDate, episodeNumber, duration})
- (:Segment {segmentId, topic, startChar, endChar})
- (:Utterance {utteranceId, text, tokenCount, globalIndex})
- (:Person {personId, name, aliases, historicalRole, description})
- (:Organization {orgId, name, aliases, orgType, description})
- (:Topic {topicId, name, category, description})
- (:Event {eventId, name, eventType, description})
- (:Claim {claimId, text, claimType, stance, confidence, status, targetDate, targetDescription})
- (:Community {communityId, level, summary, memberCount})

RELATIONSHIP TYPES:
- (:Episode)-[:HAS_SEGMENT {order}]->(:Segment)
- (:Segment)-[:HAS_CHUNK {order}]->(:Utterance)
- (:Segment)-[:NEXT_SEGMENT]->(:Segment)
- (:Utterance)-[:NEXT_CHUNK]->(:Utterance)
- (:Utterance)-[:MENTIONS]->(:Person | :Organization | :Topic | :Event)
- (:Person)-[:ROLE_IN {title, start_date, end_date}]->(:Organization)
- (:Person | :Organization)-[:PARTICIPATED_IN {outcome}]->(:Event)
- (:Event)-[:LED_TO | :CAUSED]->(:Event)
- (:Organization)-[:CONTROLLED | :LOCATED_IN]->(:Topic)
- (:Claim)-[:EXTRACTED_FROM]->(:Utterance)
- (:Claim)-[:STATED_DURING]->(:Episode)
- (:Claim)-[:ABOUT]->(:Person | :Organization | :Topic | :Event)
- (:Claim)-[:SUPPORTS {strength}]->(:Claim)
- (:Claim)-[:CONTRADICTS {strength}]->(:Claim)
- (:Claim)-[:REVISES_PREDICTION]->(:Claim)
- (:Person | :Organization | :Topic | :Event)-[:IN_COMMUNITY]->(:Community)
- (:Community)-[:PARENT_COMMUNITY]->(:Community)

PROPERTY VALUES:
- claimType: factual_assertion | historical_interpretation | geopolitical_prediction | causal_analysis
- stance: critical | favorable | neutral | speculative
- status: active | superseded | verified | refuted | debated
- historicalRole: ruler | military_leader | philosopher | diplomat | civilian | religious_leader | scientist | artist
- orgType: empire | nation-state | dynasty | alliance | religion | institution
- category (Topic): geography | ideology | economic_system | governing_system | military_strategy | religious_practice
- eventType: war | battle | treaty | revolution | crisis | period
"""

# =============================================================================
# TEXT-TO-CYPHER PROMPT
# =============================================================================

TEXT2CYPHER_SYSTEM = f"""You are a Neo4j Cypher expert. Generate READ-ONLY Cypher queries to answer questions about a history lecture knowledge graph.

GRAPH SCHEMA:
{GRAPH_SCHEMA}

RULES:
1. Generate ONLY read queries (MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT, CALL).
2. NEVER use CREATE, MERGE, DELETE, SET, REMOVE, or any write operations.
3. Always LIMIT results to at most 50 rows.
4. Use parameterized values where appropriate ($param syntax).
5. Use toLower() for case-insensitive string matching.
6. Prefer CONTAINS over exact match for name searches (handles partial names).
7. Return meaningful column names (use AS aliases).

FEW-SHOT EXAMPLES:

Q: "How many episodes discuss the Roman Empire?"
MATCH (ep:Episode)-[:HAS_SEGMENT]->(s:Segment)-[:HAS_CHUNK]->(u:Utterance)-[:MENTIONS]->(e)
WHERE e.name CONTAINS 'Roman Empire' OR 'Roman Empire' IN e.aliases
RETURN count(DISTINCT ep) AS episodeCount

Q: "What are all predictions and their status?"
MATCH (c:Claim {{claimType: 'geopolitical_prediction'}})
OPTIONAL MATCH (c)-[:STATED_DURING]->(ep:Episode)
RETURN c.text AS prediction, c.status AS status, c.targetDate AS targetDate, ep.title AS episode
ORDER BY ep.publishDate
LIMIT 50

Q: "Show causal chains involving World War I"
MATCH path = (e1:Event)-[:LED_TO|CAUSED*1..3]->(e2:Event)
WHERE e1.name CONTAINS 'World War I' OR e2.name CONTAINS 'World War I'
RETURN [n IN nodes(path) | n.name] AS causalChain
LIMIT 20

Q: "Which entities are mentioned most frequently?"
MATCH (u:Utterance)-[:MENTIONS]->(e)
RETURN labels(e)[0] AS type, e.name AS entity, count(u) AS mentions
ORDER BY mentions DESC
LIMIT 20"""

TEXT2CYPHER_USER = """Question: {question}

Generate a Cypher query to answer this question."""

TEXT2CYPHER_RETRY = """The previous Cypher query failed with this error:
{error}

Original question: {question}

Generate a CORRECTED Cypher query. Pay attention to the error message and fix the issue."""


# =============================================================================
# SAFETY CHECK
# =============================================================================

WRITE_KEYWORDS = re.compile(
    r'\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+gds)\b',
    re.IGNORECASE
)


def is_read_only(cypher: str) -> bool:
    """
    Check that a generated Cypher query is read-only.
    
    We NEVER execute write queries from Text-to-Cypher.
    This is a safety net in case the LLM ignores our instructions.
    """
    return not bool(WRITE_KEYWORDS.search(cypher))


# =============================================================================
# TEXT-TO-CYPHER WITH SELF-HEALING
# =============================================================================


async def text_to_cypher(
    graph_client: GraphClient,
    openai_client,
    question: str,
    max_retries: int = 2,
) -> dict:
    """
    Translate a natural language question to Cypher, execute it, and return results.
    
    FLOW:
    1. Send question + schema + few-shot examples to LLM
    2. LLM generates a Cypher query
    3. Validate it's read-only (safety check)
    4. Execute against Neo4j
    5. If it fails: feed error back to LLM, retry (up to max_retries)
    6. Return results or error
    
    Parameters:
        graph_client:   GraphClient instance
        openai_client:  OpenAI AsyncClient
        question:       The user's natural language question
        max_retries:    How many times to retry on failure (default: 2)
    
    Returns:
        Dict with:
        - "success": bool
        - "query": the Cypher query that was executed
        - "results": list of result rows (if success)
        - "error": error message (if failure after all retries)
    """
    last_error = None

    for attempt in range(1 + max_retries):
        # Generate Cypher
        if attempt == 0:
            # First attempt: normal prompt
            user_prompt = TEXT2CYPHER_USER.format(question=question)
        else:
            # Retry: include the error for self-healing
            user_prompt = TEXT2CYPHER_RETRY.format(
                error=last_error,
                question=question,
            )

        response = await openai_client.chat.completions.create(
            model=settings.model_cheap,  # gpt-4o-mini (fast, good at Cypher)
            temperature=0.0,
            messages=[
                {"role": "system", "content": TEXT2CYPHER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Extract the Cypher from the response
        cypher = response.choices[0].message.content.strip()

        # Clean up: remove markdown code fences if present
        if cypher.startswith("```"):
            cypher = cypher.split("\n", 1)[1] if "\n" in cypher else cypher[3:]
        if cypher.endswith("```"):
            cypher = cypher[:-3]
        cypher = cypher.strip()

        # Safety check: ensure it's read-only
        if not is_read_only(cypher):
            last_error = "SAFETY VIOLATION: Generated query contains write operations. Only READ queries are allowed."
            continue

        # Execute the query
        try:
            results = await graph_client.run(cypher)
            return {
                "success": True,
                "query": cypher,
                "results": results[:50],  # Cap at 50 rows
                "attempts": attempt + 1,
            }
        except Exception as e:
            last_error = str(e)
            # Will retry on next iteration

    # All retries exhausted
    return {
        "success": False,
        "query": cypher if 'cypher' in dir() else None,
        "results": [],
        "error": last_error,
        "attempts": 1 + max_retries,
    }
