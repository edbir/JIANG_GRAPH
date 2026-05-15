"""
Graph Upsert — Idempotent Data Insertion into Neo4j.
=====================================================

PURPOSE:
    Takes extracted knowledge (entities, relationships, claims) and inserts
    it into the Neo4j graph using MERGE patterns. Running the same data
    twice produces the same graph — no duplicates, no corruption.

THE GOLDEN RULE OF MERGE:
    NEVER merge an entire pattern with unbound variables.
    ALWAYS bind nodes first (MATCH or MERGE by ID), THEN merge relationships.
    
    ❌ MERGE (a:Person)-[:KNOWS]->(b:Person)  — creates duplicate nodes!
    ✅ MATCH (a:Person {id: $id1}) MATCH (b:Person {id: $id2}) MERGE (a)-[:KNOWS]->(b)

INSERTION ORDER:
    1. Episode (container)
    2. Segments (belong to episode)
    3. Utterances/Chunks (belong to segments)
    4. Entities (Person, Organization, Topic, Event)
    5. Entity→Utterance MENTIONS relationships
    6. Claims (extracted from utterances, reference entities)
    7. Entity↔Entity relationships (ROLE_IN, PARTICIPATED_IN, etc.)
    8. Claim↔Claim relationships (SUPPORTS, CONTRADICTS, REVISES_PREDICTION)

    This order ensures that when we create a relationship, both endpoints
    already exist in the graph.

IDEMPOTENCY:
    Every node has a deterministic ID. MERGE finds existing nodes by ID.
    ON CREATE SET: properties set only on first insertion.
    ON MATCH SET: properties updated on subsequent runs (enrichment).
"""

import hashlib
import re
import unicodedata

from src.graph.neo4j_client import GraphClient
from src.preprocessing.chunker import Chunk, Segment
from src.extraction.schemas import (
    Entity,
    EntityExtractionResult,
    Relationship,
    RelationshipExtractionResult,
    Claim,
    ClaimExtractionResult,
)


# =============================================================================
# ID GENERATION
# =============================================================================
# Deterministic IDs ensure that MERGE works correctly.
# Same input → same ID → same node (no duplicates).
# =============================================================================


def _normalize_for_id(name: str) -> str:
    """
    Normalize a name into a URL-safe, deterministic ID component.
    
    "Alexander the Great" → "alexander-the-great"
    "Roman Empire" → "roman-empire"
    "Battle of Thermopylae" → "battle-of-thermopylae"
    """
    # Lowercase
    text = name.lower()
    # Remove accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Replace non-alphanumeric with hyphens
    text = re.sub(r'[^a-z0-9]+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text


def make_entity_id(entity_type: str, canonical_name: str) -> str:
    """
    Generate a deterministic ID for an entity.
    
    Examples:
        make_entity_id("person", "Alexander the Great") → "person-alexander-the-great"
        make_entity_id("organization", "Roman Empire") → "org-roman-empire"
    """
    prefix_map = {
        "person": "person",
        "organization": "org",
        "topic": "topic",
        "event": "event",
    }
    prefix = prefix_map.get(entity_type, entity_type)
    return f"{prefix}-{_normalize_for_id(canonical_name)}"


def make_claim_id(claim_text: str) -> str:
    """
    Generate a deterministic ID for a claim based on its text content.
    
    Uses SHA-256 hash truncated to 12 chars. Same claim text → same ID.
    
    Example:
        make_claim_id("The fall of Rome was caused by...") → "claim-a3f2b9c1d4e5"
    """
    text_hash = hashlib.sha256(claim_text.encode()).hexdigest()[:12]
    return f"claim-{text_hash}"


# =============================================================================
# CONTENT LAYER UPSERT (Episode → Segment → Utterance)
# =============================================================================


async def upsert_episode(
    client: GraphClient,
    episode_id: str,
    title: str,
    episode_number: int,
    publish_date: str = None,
    duration: str = None,
) -> None:
    """
    Insert or update an Episode node.
    
    This is the top of the content hierarchy.
    All segments and utterances belong to an episode.
    """
    await client.run("""
        MERGE (e:Episode {episodeId: $episodeId})
        ON CREATE SET
            e.title = $title,
            e.episodeNumber = $episodeNumber,
            e.publishDate = $publishDate,
            e.duration = $duration
        ON MATCH SET
            e.title = $title
    """, {
        "episodeId": episode_id,
        "title": title,
        "episodeNumber": episode_number,
        "publishDate": publish_date,
        "duration": duration,
    })


async def upsert_segments(
    client: GraphClient,
    episode_id: str,
    segments: list[Segment],
) -> None:
    """
    Insert Segment nodes and connect them to their Episode.
    Also creates NEXT_SEGMENT chain for sequential ordering.
    """
    for segment in segments:
        segment_id = f"{episode_id}-seg{segment.segment_index:02d}"

        # Create/update the segment node
        await client.run("""
            MERGE (s:Segment {segmentId: $segmentId})
            ON CREATE SET
                s.topic = $topic,
                s.startChar = $startChar,
                s.endChar = $endChar
            ON MATCH SET
                s.topic = $topic
        """, {
            "segmentId": segment_id,
            "topic": segment.topic,
            "startChar": segment.start_char,
            "endChar": segment.end_char,
        })

        # Connect segment to episode
        await client.run("""
            MATCH (e:Episode {episodeId: $episodeId})
            MATCH (s:Segment {segmentId: $segmentId})
            MERGE (e)-[:HAS_SEGMENT {order: $order}]->(s)
        """, {
            "episodeId": episode_id,
            "segmentId": segment_id,
            "order": segment.segment_index,
        })

    # Create NEXT_SEGMENT chain (sequential ordering)
    for i in range(len(segments) - 1):
        current_id = f"{episode_id}-seg{segments[i].segment_index:02d}"
        next_id = f"{episode_id}-seg{segments[i + 1].segment_index:02d}"

        await client.run("""
            MATCH (s1:Segment {segmentId: $currentId})
            MATCH (s2:Segment {segmentId: $nextId})
            MERGE (s1)-[:NEXT_SEGMENT]->(s2)
        """, {"currentId": current_id, "nextId": next_id})


async def upsert_utterances(
    client: GraphClient,
    chunks: list[Chunk],
) -> None:
    """
    Insert Utterance (chunk) nodes and connect them to their Segments.
    Also creates NEXT_CHUNK chain for sequential ordering.
    """
    for chunk in chunks:
        segment_id = f"{chunk.episode_id}-seg{chunk.segment_index:02d}"

        # Create/update the utterance node
        await client.run("""
            MERGE (u:Utterance {utteranceId: $utteranceId})
            ON CREATE SET
                u.text = $text,
                u.tokenCount = $tokenCount,
                u.globalIndex = $globalIndex,
                u.startChar = $startChar,
                u.endChar = $endChar
            ON MATCH SET
                u.text = $text
        """, {
            "utteranceId": chunk.chunk_id,
            "text": chunk.text,
            "tokenCount": chunk.token_count,
            "globalIndex": chunk.global_index,
            "startChar": chunk.start_char,
            "endChar": chunk.end_char,
        })

        # Connect utterance to segment
        await client.run("""
            MATCH (s:Segment {segmentId: $segmentId})
            MATCH (u:Utterance {utteranceId: $utteranceId})
            MERGE (s)-[:HAS_CHUNK {order: $order}]->(u)
        """, {
            "segmentId": segment_id,
            "utteranceId": chunk.chunk_id,
            "order": chunk.chunk_index,
        })

    # Create NEXT_CHUNK chain
    for i in range(len(chunks) - 1):
        await client.run("""
            MATCH (u1:Utterance {utteranceId: $currentId})
            MATCH (u2:Utterance {utteranceId: $nextId})
            MERGE (u1)-[:NEXT_CHUNK]->(u2)
        """, {
            "currentId": chunks[i].chunk_id,
            "nextId": chunks[i + 1].chunk_id,
        })


# =============================================================================
# ENTITY LAYER UPSERT (Person, Organization, Topic, Event)
# =============================================================================


async def upsert_entity(
    client: GraphClient,
    entity: Entity,
    source_chunk_id: str,
    merge_map: dict[str, str] = None,
) -> str:
    """
    Insert or update an entity node and connect it to its source utterance.
    
    Applies the merge_map to use canonical names (from entity resolution).
    Returns the entity ID used (for relationship creation later).
    
    ENRICHMENT LOGIC:
        If the entity already exists, we keep the LONGER description
        (usually more informative) and increment sourceCount.
    """
    # Apply merge map to get canonical name
    canonical_name = entity.name
    if merge_map and entity.name in merge_map:
        canonical_name = merge_map[entity.name]

    entity_id = make_entity_id(entity.entity_type.value, canonical_name)

    # Determine the label based on entity type
    label_map = {
        "person": "Person",
        "organization": "Organization",
        "topic": "Topic",
        "event": "Event",
    }
    label = label_map[entity.entity_type.value]

    # Determine the ID field name based on type
    id_field_map = {
        "Person": "personId",
        "Organization": "orgId",
        "Topic": "topicId",
        "Event": "eventId",
    }
    id_field = id_field_map[label]

    # Build properties based on entity type
    properties = {
        "entityId": entity_id,
        "name": canonical_name,
        "description": entity.description,
        "aliases": entity.aliases,
    }

    # Add type-specific properties
    if entity.entity_type.value == "person" and entity.historical_role:
        properties["historicalRole"] = entity.historical_role
    elif entity.entity_type.value == "organization" and entity.org_type:
        properties["orgType"] = entity.org_type
    elif entity.entity_type.value == "topic" and entity.topic_category:
        properties["category"] = entity.topic_category
    elif entity.entity_type.value == "event" and entity.event_type:
        properties["eventType"] = entity.event_type

    # MERGE the entity node (using the type-specific ID field)
    await client.run(f"""
        MERGE (e:{label} {{{id_field}: $entityId}})
        ON CREATE SET
            e.name = $name,
            e.description = $description,
            e.aliases = $aliases,
            e.sourceCount = 1
        ON MATCH SET
            e.description = CASE
                WHEN size(e.description) < size($description)
                THEN $description ELSE e.description
            END,
            e.aliases = [x IN e.aliases + $aliases WHERE x <> e.name | x],
            e.sourceCount = e.sourceCount + 1
    """, properties)

    # Set type-specific properties (separate query for clarity)
    if "historicalRole" in properties:
        await client.run(f"""
            MATCH (e:{label} {{{id_field}: $entityId}})
            SET e.historicalRole = $role
        """, {"entityId": entity_id, "role": properties["historicalRole"]})
    elif "orgType" in properties:
        await client.run(f"""
            MATCH (e:{label} {{{id_field}: $entityId}})
            SET e.orgType = $orgType
        """, {"entityId": entity_id, "orgType": properties["orgType"]})
    elif "category" in properties:
        await client.run(f"""
            MATCH (e:{label} {{{id_field}: $entityId}})
            SET e.category = $category
        """, {"entityId": entity_id, "category": properties["category"]})
    elif "eventType" in properties:
        await client.run(f"""
            MATCH (e:{label} {{{id_field}: $entityId}})
            SET e.eventType = $eventType
        """, {"entityId": entity_id, "eventType": properties["eventType"]})

    # Create MENTIONS relationship (Utterance → Entity)
    await client.run(f"""
        MATCH (u:Utterance {{utteranceId: $chunkId}})
        MATCH (e:{label} {{{id_field}: $entityId}})
        MERGE (u)-[:MENTIONS]->(e)
    """, {"chunkId": source_chunk_id, "entityId": entity_id})

    return entity_id


# =============================================================================
# CLAIM LAYER UPSERT
# =============================================================================


async def upsert_claim(
    client: GraphClient,
    claim: Claim,
    source_chunk_id: str,
    episode_id: str,
    merge_map: dict[str, str] = None,
) -> str:
    """
    Insert a Claim node and connect it to its source utterance, episode, and entities.
    
    Claims are first-class nodes (reification pattern). They connect to:
    - The Utterance they were extracted from (EXTRACTED_FROM)
    - The Episode they were stated during (STATED_DURING)
    - The entities they reference (ABOUT)
    """
    claim_id = make_claim_id(claim.text)

    # Build claim properties
    properties: dict = {
        "claimId": claim_id,
        "text": claim.text,
        "claimType": claim.claim_type.value,
        "stance": claim.stance.value,
        "confidence": claim.confidence,
        "status": "active",  # Default status; predictions may become verified/refuted
    }

    # Add prediction-specific fields
    if claim.target_date:
        properties["targetDate"] = claim.target_date
    if claim.target_description:
        properties["targetDescription"] = claim.target_description

    # MERGE the claim node
    await client.run("""
        MERGE (c:Claim {claimId: $claimId})
        ON CREATE SET
            c.text = $text,
            c.claimType = $claimType,
            c.stance = $stance,
            c.confidence = $confidence,
            c.status = $status
    """, properties)

    # Set optional prediction fields if present
    if claim.target_date:
        await client.run("""
            MATCH (c:Claim {claimId: $claimId})
            SET c.targetDate = $targetDate
        """, {"claimId": claim_id, "targetDate": claim.target_date})

    if claim.target_description:
        await client.run("""
            MATCH (c:Claim {claimId: $claimId})
            SET c.targetDescription = $targetDescription
        """, {"claimId": claim_id, "targetDescription": claim.target_description})

    # Connect claim to source utterance (provenance)
    await client.run("""
        MATCH (c:Claim {claimId: $claimId})
        MATCH (u:Utterance {utteranceId: $chunkId})
        MERGE (c)-[:EXTRACTED_FROM]->(u)
    """, {"claimId": claim_id, "chunkId": source_chunk_id})

    # Connect claim to episode (temporal anchor)
    await client.run("""
        MATCH (c:Claim {claimId: $claimId})
        MATCH (e:Episode {episodeId: $episodeId})
        MERGE (c)-[:STATED_DURING]->(e)
    """, {"claimId": claim_id, "episodeId": episode_id})

    # Connect claim to referenced entities (ABOUT)
    for entity_name in claim.entities_involved:
        # Apply merge map
        canonical = entity_name
        if merge_map and entity_name in merge_map:
            canonical = merge_map[entity_name]

        # Try to find the entity by name (across all entity types)
        await client.run("""
            MATCH (c:Claim {claimId: $claimId})
            OPTIONAL MATCH (e) WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
                AND e.name = $entityName
            WITH c, e WHERE e IS NOT NULL
            MERGE (c)-[:ABOUT]->(e)
        """, {"claimId": claim_id, "entityName": canonical})

    return claim_id


# =============================================================================
# RELATIONSHIP UPSERT (Entity ↔ Entity)
# =============================================================================


async def upsert_relationship(
    client: GraphClient,
    relationship: Relationship,
    merge_map: dict[str, str] = None,
) -> None:
    """
    Insert a relationship between two entities.
    
    Both entities must already exist in the graph (inserted via upsert_entity).
    Uses the canonical names from the merge map.
    """
    # Apply merge map
    source_name = relationship.source_entity
    target_name = relationship.target_entity
    if merge_map:
        source_name = merge_map.get(source_name, source_name)
        target_name = merge_map.get(target_name, target_name)

    rel_type = relationship.relationship_type.value

    # Find source and target by name, then MERGE the relationship
    # We search across all entity types since we don't know the label here
    await client.run(f"""
        MATCH (source) WHERE (source:Person OR source:Organization OR source:Topic OR source:Event)
            AND source.name = $sourceName
        MATCH (target) WHERE (target:Person OR target:Organization OR target:Topic OR target:Event)
            AND target.name = $targetName
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET r.description = $description, r.confidence = $confidence, r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
    """, {
        "sourceName": source_name,
        "targetName": target_name,
        "description": relationship.description,
        "confidence": relationship.confidence,
    })


# =============================================================================
# EMBEDDING STORAGE
# =============================================================================


async def store_embedding(
    client: GraphClient,
    label: str,
    id_field: str,
    node_id: str,
    embedding: list[float],
) -> None:
    """
    Store an embedding vector on an existing node.
    
    Called after all nodes are inserted, in a separate embedding generation pass.
    
    Parameters:
        label:     Node label (e.g., "Utterance", "Claim", "Person")
        id_field:  The ID property name (e.g., "utteranceId", "claimId", "personId")
        node_id:   The actual ID value
        embedding: The embedding vector (list of 1536 floats)
    """
    await client.run(f"""
        MATCH (n:{label} {{{id_field}: $nodeId}})
        SET n.embedding = $embedding
    """, {"nodeId": node_id, "embedding": embedding})


# =============================================================================
# FULL EPISODE UPSERT (orchestrates everything)
# =============================================================================


async def upsert_episode_data(
    client: GraphClient,
    episode_id: str,
    episode_title: str,
    episode_number: int,
    segments: list[Segment],
    chunks: list[Chunk],
    extraction_results: dict,
    merge_map: dict[str, str] = None,
) -> dict:
    """
    Insert all data for a single episode into the graph.
    
    This is the main entry point for the upsert module.
    It orchestrates the insertion in the correct order.
    
    Parameters:
        client:             GraphClient instance
        episode_id:         Episode identifier
        episode_title:      Title of the episode
        episode_number:     Episode number
        segments:           Level 1 segments from chunker
        chunks:             Level 2 chunks from chunker
        extraction_results: Output of extractor.process_episode()
        merge_map:          Entity resolution merge map (from resolver)
    
    Returns:
        Dict with counts of inserted nodes/relationships for monitoring
    """
    if merge_map is None:
        merge_map = {}

    counts = {
        "entities": 0,
        "relationships": 0,
        "claims": 0,
        "mentions": 0,
    }

    # --- Step 1: Content hierarchy ---
    await upsert_episode(client, episode_id, episode_title, episode_number)
    await upsert_segments(client, episode_id, segments)
    await upsert_utterances(client, chunks)

    # --- Step 2: Entities (from all chunks) ---
    entities_data = extraction_results.get("entities", {})
    for chunk_id, entity_result in entities_data.items():
        for entity in entity_result.entities:
            await upsert_entity(client, entity, chunk_id, merge_map)
            counts["entities"] += 1

    # --- Step 3: Claims (from all chunks) ---
    claims_data = extraction_results.get("claims", {})
    for chunk_id, claim_result in claims_data.items():
        for claim in claim_result.claims:
            # Only ingest claims above confidence threshold
            if claim.confidence >= 0.7:
                await upsert_claim(client, claim, chunk_id, episode_id, merge_map)
                counts["claims"] += 1

    # --- Step 4: Entity relationships (from all chunks) ---
    relationships_data = extraction_results.get("relationships", {})
    for chunk_id, rel_result in relationships_data.items():
        for relationship in rel_result.relationships:
            if relationship.confidence >= 0.7:
                await upsert_relationship(client, relationship, merge_map)
                counts["relationships"] += 1

    return counts
