"""
Query Classifier — Routes queries to optimal retrieval strategies.
===================================================================

PURPOSE:
    Takes a user's natural language question and determines:
    1. What TYPE of query it is (factual, comparative, temporal, thematic)
    2. What ENTITIES are mentioned (for filtering)
    3. What TIME RANGE is referenced (if any)
    4. What TOPICS are involved

    This classification drives the Strategy Router — each query type
    gets a different retrieval approach optimized for that pattern.

WHY CLASSIFY (not just vector search everything):
    - "What did Jiang say about Rome?" → vector search works great
    - "How do Caesar and Alexander compare?" → needs structured claim retrieval
    - "How has the prediction about the US evolved?" → needs chronological ordering
    - "What are the main themes?" → needs community summaries, not passages
    
    One strategy cannot serve all these well. Classification enables routing.

REFERENCE:
    Adaptive-RAG (Jeong et al., NAACL 2024) is the seminal paper on
    query-adaptive retrieval. Our approach is simpler (4 fixed categories
    vs. their learned complexity predictor) but follows the same principle.
"""

from pydantic import BaseModel, Field
from enum import Enum

from src.config import settings


# =============================================================================
# CLASSIFICATION SCHEMA
# =============================================================================


class QueryType(str, Enum):
    """The four query types our system supports."""
    FACTUAL = "factual"           # "What did Jiang say about X?"
    COMPARATIVE = "comparative"   # "How do A and B differ on X?"
    TEMPORAL = "temporal"         # "How has the view on X evolved?"
    THEMATIC = "thematic"        # "What are the main themes?"


class QueryClassification(BaseModel):
    """
    Output of the query classifier.
    Drives the strategy router to pick the right retrieval approach.
    """
    query_type: QueryType = Field(
        description="The type of query, determining which retrieval strategy to use."
    )
    mentioned_entities: list[str] = Field(
        default_factory=list,
        description="Entity names mentioned in the query (persons, orgs, topics, events)."
    )
    time_range: str | None = Field(
        default=None,
        description="Time period referenced, if any. E.g., '2024', 'ancient Rome', 'last 5 episodes'."
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topics or themes the query is about."
    )


# =============================================================================
# CLASSIFICATION PROMPT
# =============================================================================

CLASSIFIER_SYSTEM = """You are classifying a user's question about a history lecture series ("Predictive History" by Prof. Jiang).

Classify the query into ONE of four types:

1. **factual** — The user wants to know what Jiang said about a specific topic or entity.
   Examples: "What did Jiang say about the fall of Rome?", "Tell me about Alexander's conquests"

2. **comparative** — The user wants to compare entities, claims, or perspectives.
   Examples: "How do Caesar and Alexander differ as leaders?", "Compare Rome and Persia"

3. **temporal** — The user wants to see how something evolved over time across episodes.
   Examples: "How has Jiang's prediction about the US changed?", "Track claims about democracy over time"

4. **thematic** — The user wants broad themes, summaries, or overviews across all content.
   Examples: "What are the main themes?", "Give me an overview of all topics covered"

Also extract:
- mentioned_entities: Any specific persons, organizations, events, or topics named in the query
- time_range: Any time period referenced (historical era, date range, or episode range)
- topics: General topics or themes the query is about"""

CLASSIFIER_USER = """Query: {query}

Classify this query."""


# =============================================================================
# CLASSIFIER FUNCTION
# =============================================================================


async def classify_query(
    client,
    query: str,
) -> QueryClassification:
    """
    Classify a user query into one of four types.
    
    Uses GPT-4o-mini with structured output for fast, cheap classification.
    This is the FIRST step in every retrieval request — it determines
    which strategies to activate.
    
    Parameters:
        client: OpenAI AsyncClient
        query:  The user's natural language question
    
    Returns:
        QueryClassification with type, entities, time_range, topics
    """
    import json

    user_prompt = CLASSIFIER_USER.format(query=query)

    response = await client.chat.completions.create(
        model=settings.model_cheap,  # gpt-4o-mini (fast, cheap)
        temperature=0.0,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "QueryClassification",
                "strict": True,
                "schema": QueryClassification.model_json_schema(),
            }
        },
    )

    result = json.loads(response.choices[0].message.content)
    return QueryClassification(**result)
