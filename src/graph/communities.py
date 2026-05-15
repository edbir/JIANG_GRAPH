"""
Community Detection and Summarization.
=======================================

PURPOSE:
    Detect thematic clusters in the knowledge graph using the Leiden algorithm,
    then generate human-readable summaries for each cluster using an LLM.

WHAT ARE COMMUNITIES:
    Communities are groups of entities (Person, Organization, Topic, Event) that
    are densely connected to each other. They represent THEMES in the lectures.
    
    Example community: {Alexander the Great, Philip II, Macedonia, Persian Empire,
    Battle of Gaugamela, Battle of Issus, Hellenistic Period}
    → Summary: "Alexander's conquest of Persia and the spread of Hellenistic culture"

WHY THIS MATTERS:
    - Enables thematic queries: "What are the main themes across all episodes?"
    - Provides high-level context for retrieval (community summaries as context)
    - Validates extraction quality (nonsensical communities = extraction problems)

THE PROCESS:
    1. Project an in-memory graph (entities + their relationships) via Neo4j GDS
    2. Run Leiden algorithm → assigns each entity to a community at multiple levels
    3. Write community IDs back to entity nodes
    4. For each community: gather members, generate LLM summary, store as Community node

DEPENDENCIES:
    - Neo4j GDS plugin (included in docker-compose.yml)
    - OpenAI API (for summarization)

INCREMENTALITY:
    When new episodes are added:
    - Re-run Leiden with seedProperty (reuses previous assignments as starting point)
    - Only re-summarize communities whose membership changed
    - For 1 new episode: typically 5-15 communities need re-summarization
"""

import json

from src.graph.neo4j_client import GraphClient
from src.config import settings


# =============================================================================
# COMMUNITY SUMMARIZATION PROMPT
# =============================================================================
# This prompt asks the LLM to describe what a community of entities represents.
# It lives here (not in prompts.py) because it's specific to community detection,
# not to the extraction pipeline.
# =============================================================================

COMMUNITY_SUMMARY_SYSTEM = """You are summarizing a thematic cluster of entities from a history lecture series ("Predictive History" by Prof. Jiang).

You will receive a list of entities (persons, organizations, topics, events) that are densely connected in a knowledge graph. Your job is to write a concise summary explaining:
1. What theme or topic unifies these entities
2. The key figures and their roles
3. The main events or concepts involved
4. Why these entities cluster together (what connects them)

Write 2-4 sentences. Be specific and informative. This summary will be used to help users find relevant content when they ask thematic questions."""

COMMUNITY_SUMMARY_USER = """COMMUNITY MEMBERS:
{members_text}

INTERNAL RELATIONSHIPS:
{relationships_text}

Summarize what this community represents."""


# =============================================================================
# GRAPH PROJECTION
# =============================================================================
# Before running Leiden, we need to "project" a subgraph into GDS memory.
# This means: tell GDS which nodes and relationships to consider.
#
# We project entity nodes (Person, Org, Topic, Event) and their relationships.
# We do NOT include content nodes (Episode, Segment, Utterance) because
# communities should represent THEMATIC clusters, not structural ones.
# =============================================================================


async def project_entity_graph(
    client: GraphClient,
    graph_name: str = "entity-graph",
) -> dict:
    """
    Project the entity co-occurrence graph into GDS memory.
    
    WHAT GETS PROJECTED:
        Nodes: Person, Organization, Topic, Event
        Relationships: All relationships between these entities
            (ROLE_IN, PARTICIPATED_IN, LED_TO, CAUSED, CONTROLLED, LOCATED_IN, ABOUT)
        
        Relationships are treated as UNDIRECTED for community detection
        (direction doesn't matter for clustering — only connectivity).
    
    WHY IN-MEMORY:
        GDS algorithms run on an in-memory graph projection, not directly
        on the stored graph. This is faster and allows us to configure
        which parts of the graph to analyze.
    
    Returns:
        Dict with projection metadata (nodeCount, relationshipCount)
    """
    # First, drop existing projection if it exists (for re-runs)
    try:
        await client.run(f"CALL gds.graph.drop('{graph_name}', false)")
    except Exception:
        pass  # Graph didn't exist, that's fine

    # Project the entity graph
    # We use a Cypher projection for flexibility
    result = await client.run(f"""
        CALL gds.graph.project(
            '{graph_name}',
            ['Person', 'Organization', 'Topic', 'Event'],
            {{
                ROLE_IN: {{orientation: 'UNDIRECTED'}},
                PARTICIPATED_IN: {{orientation: 'UNDIRECTED'}},
                LED_TO: {{orientation: 'UNDIRECTED'}},
                CAUSED: {{orientation: 'UNDIRECTED'}},
                CONTROLLED: {{orientation: 'UNDIRECTED'}},
                LOCATED_IN: {{orientation: 'UNDIRECTED'}},
                ABOUT: {{orientation: 'UNDIRECTED'}}
            }}
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
    """)

    if result:
        return result[0]
    return {"graphName": graph_name, "nodeCount": 0, "relationshipCount": 0}


# =============================================================================
# LEIDEN COMMUNITY DETECTION
# =============================================================================


async def run_leiden(
    client: GraphClient,
    graph_name: str = "entity-graph",
    use_seed: bool = False,
    gamma: float = 1.0,
) -> dict:
    """
    Run the Leiden algorithm on the projected entity graph.
    
    WHAT LEIDEN DOES:
        Partitions the graph into communities by optimizing modularity.
        Modularity measures: "are nodes within a community more connected
        to each other than to nodes outside?" High modularity = good partition.
    
    HIERARCHICAL OUTPUT:
        With includeIntermediateCommunities=true, Leiden returns communities
        at ALL levels of the hierarchy:
        - Level 0: Fine-grained clusters (5-20 entities)
        - Level 1: Medium clusters (20-50 entities)
        - Level 2+: Broad themes (50+ entities)
    
    PARAMETERS:
        gamma: Resolution parameter. Higher = more, smaller communities.
               Default 1.0 is standard. Increase to 1.5-2.0 if communities
               are too large and mix unrelated topics.
        
        use_seed: If True, uses existing communityId as starting point.
                  This makes incremental updates efficient — Leiden starts
                  from the previous partition instead of from scratch.
    
    Returns:
        Dict with communityCount, modularity, ranLevels
    """
    # Build the Leiden configuration
    config = {
        "writeProperty": "communityId",
        "includeIntermediateCommunities": True,
        "gamma": gamma,
        "randomSeed": 42,  # Reproducibility
    }

    # If using seed (incremental update), add seedProperty
    if use_seed:
        config["seedProperty"] = "communityId"

    # Convert config to Cypher map syntax
    config_str = json.dumps(config).replace('"', "'")
    # Fix booleans for Cypher (Python True → Cypher true)
    config_str = config_str.replace("'True'", "true").replace("'False'", "false")
    config_str = config_str.replace("True", "true").replace("False", "false")

    result = await client.run(f"""
        CALL gds.leiden.write('{graph_name}', {{
            writeProperty: 'communityId',
            includeIntermediateCommunities: true,
            gamma: {gamma},
            randomSeed: 42
            {", seedProperty: 'communityId'" if use_seed else ""}
        }})
        YIELD communityCount, modularity, ranLevels
        RETURN communityCount, modularity, ranLevels
    """)

    if result:
        return result[0]
    return {"communityCount": 0, "modularity": 0.0, "ranLevels": 0}


# =============================================================================
# COMMUNITY MEMBER GATHERING
# =============================================================================


async def get_community_members(
    client: GraphClient,
    community_id: int,
    level: int = 0,
) -> dict:
    """
    Gather all members and internal relationships of a community.
    
    This data is what we pass to the LLM for summarization.
    
    Returns:
        Dict with:
        - members: list of {name, labels, description}
        - relationships: list of {source, type, target, description}
    """
    # Get member entities
    members_result = await client.run("""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
            AND e.communityId = $communityId
        RETURN e.name AS name, labels(e) AS labels, e.description AS description
        ORDER BY e.sourceCount DESC
        LIMIT 50
    """, {"communityId": community_id})

    # Get internal relationships (between members of the same community)
    rels_result = await client.run("""
        MATCH (a)-[r]->(b)
        WHERE (a:Person OR a:Organization OR a:Topic OR a:Event)
            AND (b:Person OR b:Organization OR b:Topic OR b:Event)
            AND a.communityId = $communityId
            AND b.communityId = $communityId
        RETURN a.name AS source, type(r) AS relType, b.name AS target,
               r.description AS description
        LIMIT 100
    """, {"communityId": community_id})

    return {
        "members": members_result,
        "relationships": rels_result,
    }


def format_community_for_prompt(community_data: dict) -> tuple[str, str]:
    """
    Format community members and relationships as text for the LLM prompt.
    
    Returns:
        Tuple of (members_text, relationships_text)
    """
    # Format members
    members_lines = []
    for member in community_data["members"]:
        labels = [l for l in member["labels"] if l not in ("__Entity__",)]
        label_str = "/".join(labels) if labels else "Entity"
        desc = member.get("description", "")
        members_lines.append(f"- {member['name']} ({label_str}): {desc}")

    members_text = "\n".join(members_lines) if members_lines else "No members found."

    # Format relationships
    rels_lines = []
    for rel in community_data["relationships"]:
        desc = rel.get("description", "")
        desc_str = f" — {desc}" if desc else ""
        rels_lines.append(f"- {rel['source']} → {rel['relType']} → {rel['target']}{desc_str}")

    relationships_text = "\n".join(rels_lines) if rels_lines else "No internal relationships found."

    return members_text, relationships_text


# =============================================================================
# COMMUNITY SUMMARIZATION
# =============================================================================


async def summarize_community(
    client: GraphClient,
    llm_client,
    community_id: int,
    level: int = 0,
) -> str:
    """
    Generate an LLM summary for a single community.
    
    FLOW:
    1. Gather community members and their internal relationships
    2. Format as text for the prompt
    3. Call GPT-4o-mini to generate a 2-4 sentence summary
    4. Return the summary text
    
    The summary is stored on the Community node by the caller.
    """
    # Gather data
    community_data = await get_community_members(client, community_id, level)

    if not community_data["members"]:
        return "Empty community — no members found."

    # Format for prompt
    members_text, relationships_text = format_community_for_prompt(community_data)

    user_prompt = COMMUNITY_SUMMARY_USER.format(
        members_text=members_text,
        relationships_text=relationships_text,
    )

    # Call LLM for summary
    response = await llm_client.chat.completions.create(
        model=settings.model_cheap,  # gpt-4o-mini is sufficient for summarization
        temperature=0.0,
        messages=[
            {"role": "system", "content": COMMUNITY_SUMMARY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    summary = response.choices[0].message.content.strip()
    return summary


# =============================================================================
# COMMUNITY NODE CREATION
# =============================================================================


async def create_community_nodes(
    client: GraphClient,
    llm_client,
    embedding_client=None,
) -> int:
    """
    Create Community nodes for all detected communities and generate summaries.
    
    FLOW:
    1. Find all unique community IDs assigned by Leiden
    2. For each community:
       a. Generate a summary via LLM
       b. Create a Community node with the summary
       c. Generate and store an embedding of the summary (for vector search)
       d. Create IN_COMMUNITY relationships from member entities
    
    Parameters:
        client:           GraphClient for Neo4j
        llm_client:       OpenAI AsyncClient for summarization
        embedding_client: OpenAI AsyncClient for embeddings (can be same as llm_client)
    
    Returns:
        Number of communities created
    """
    if embedding_client is None:
        embedding_client = llm_client

    # Find all unique community IDs
    communities = await client.run("""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
            AND e.communityId IS NOT NULL
        RETURN DISTINCT e.communityId AS communityId, count(e) AS memberCount
        ORDER BY memberCount DESC
    """)

    count = 0

    for community in communities:
        community_id = community["communityId"]
        member_count = community["memberCount"]

        # Skip tiny communities (likely noise)
        if member_count < 3:
            continue

        # Generate summary
        summary = await summarize_community(client, llm_client, community_id)

        # Create Community node
        node_id = f"community-0-{community_id}"
        await client.run("""
            MERGE (c:Community {communityId: $nodeId})
            ON CREATE SET
                c.level = 0,
                c.summary = $summary,
                c.memberCount = $memberCount,
                c.leidenId = $leidenId
            ON MATCH SET
                c.summary = $summary,
                c.memberCount = $memberCount
        """, {
            "nodeId": node_id,
            "summary": summary,
            "memberCount": member_count,
            "leidenId": community_id,
        })

        # Generate and store embedding for the summary
        embedding_response = await embedding_client.embeddings.create(
            model=settings.embedding_model,
            input=summary,
        )
        embedding = embedding_response.data[0].embedding

        await client.run("""
            MATCH (c:Community {communityId: $nodeId})
            SET c.embedding = $embedding
        """, {"nodeId": node_id, "embedding": embedding})

        # Create IN_COMMUNITY relationships from members
        await client.run("""
            MATCH (e)
            WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
                AND e.communityId = $leidenId
            MATCH (c:Community {communityId: $nodeId})
            MERGE (e)-[:IN_COMMUNITY]->(c)
        """, {"leidenId": community_id, "nodeId": node_id})

        count += 1

    return count


# =============================================================================
# FULL COMMUNITY PIPELINE
# =============================================================================


async def detect_and_summarize_communities(
    client: GraphClient,
    llm_client,
    use_seed: bool = False,
    gamma: float = 1.0,
) -> dict:
    """
    Full community detection pipeline. This is the main entry point.
    
    STEPS:
    1. Project entity graph into GDS memory
    2. Run Leiden algorithm (assigns communityId to each entity)
    3. Create Community nodes with LLM-generated summaries
    4. Drop the in-memory projection (free memory)
    
    Parameters:
        client:     GraphClient for Neo4j
        llm_client: OpenAI AsyncClient
        use_seed:   If True, use existing communities as starting point (incremental)
        gamma:      Leiden resolution parameter (higher = more communities)
    
    Returns:
        Dict with statistics: {projected, leiden, communities_created}
    """
    graph_name = "entity-graph"

    # Step 1: Project
    projection = await project_entity_graph(client, graph_name)

    # Step 2: Run Leiden
    leiden_result = await run_leiden(client, graph_name, use_seed=use_seed, gamma=gamma)

    # Step 3: Create community nodes with summaries
    communities_created = await create_community_nodes(client, llm_client)

    # Step 4: Drop projection (free GDS memory)
    try:
        await client.run(f"CALL gds.graph.drop('{graph_name}', false)")
    except Exception:
        pass

    return {
        "projected": projection,
        "leiden": leiden_result,
        "communities_created": communities_created,
    }
