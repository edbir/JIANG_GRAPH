"""
Neo4j Client — Connection, Schema Setup, and Index Management.
================================================================

PURPOSE:
    This module handles everything related to the Neo4j database itself:
    - Connecting to the database
    - Creating constraints (uniqueness guarantees)
    - Creating indexes (vector, full-text, composite)
    - Providing a clean interface for running Cypher queries

    It does NOT handle data insertion (that's upsert.py) or community
    detection (that's communities.py). It's purely about the database
    infrastructure.

WHY CONSTRAINTS AND INDEXES MATTER:
    Constraints:
        Prevent duplicate nodes. If we try to create two Episode nodes
        with the same episodeId, Neo4j will reject the second one.
        This is our safety net against the extraction pipeline
        accidentally creating duplicates.
    
    Vector Indexes:
        Enable fast similarity search over embeddings. Without these,
        a vector query would scan ALL nodes — with them, it uses
        approximate nearest neighbor (ANN) for sub-second results.
    
    Full-Text Indexes:
        Enable keyword search over text fields. Useful for finding
        chunks/claims that mention specific terms.
    
    Composite Indexes:
        Speed up common query patterns (e.g., "all claims of type X
        with status Y").

USAGE:
    from src.graph.neo4j_client import GraphClient
    
    async with GraphClient() as client:
        await client.setup_schema()
        result = await client.run("MATCH (n) RETURN count(n)")
"""

from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from src.config import settings


# =============================================================================
# THE CLIENT CLASS
# =============================================================================


class GraphClient:
    """
    Async Neo4j client with schema management.
    
    Wraps the Neo4j async driver and provides:
    - Connection lifecycle management (connect/close)
    - Schema setup (constraints + indexes)
    - A simple `run()` method for executing Cypher queries
    
    Can be used as an async context manager:
        async with GraphClient() as client:
            await client.run("MATCH (n) RETURN n LIMIT 5")
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
    ):
        """
        Initialize the client. Does NOT connect yet — call connect() or use as context manager.
        
        Parameters default to values from config.py (which reads from .env).
        """
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """
        Establish connection to Neo4j.
        
        The driver manages a connection pool internally —
        we don't need to worry about individual connections.
        """
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self):
        """Support `async with GraphClient() as client:`"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close on context manager exit."""
        await self.close()

    def _get_session(self) -> AsyncSession:
        """Get a session from the driver. Internal use."""
        if not self._driver:
            raise RuntimeError("Not connected. Call connect() first or use as context manager.")
        return self._driver.session()

    async def run(self, query: str, parameters: dict = None) -> list[dict]:
        """
        Execute a Cypher query and return results as a list of dicts.
        
        This is the main method for running arbitrary Cypher.
        For data insertion, use upsert.py which builds the queries for you.
        
        Parameters:
            query:      Cypher query string
            parameters: Dict of query parameters (use $param_name in Cypher)
        
        Returns:
            List of result records as dicts
        
        Example:
            results = await client.run(
                "MATCH (e:Episode) WHERE e.episodeNumber = $num RETURN e",
                {"num": 42}
            )
        """
        async with self._get_session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    # =========================================================================
    # SCHEMA SETUP
    # =========================================================================
    # Run this ONCE when setting up the database for the first time.
    # It's idempotent — running it again does nothing (IF NOT EXISTS).
    # =========================================================================

    async def setup_schema(self) -> None:
        """
        Create all constraints and indexes for the knowledge graph.
        
        This sets up:
        1. Uniqueness constraints (prevent duplicate nodes)
        2. Vector indexes (for semantic similarity search)
        3. Full-text indexes (for keyword search)
        4. Composite indexes (for common query patterns)
        
        Safe to run multiple times — all statements use IF NOT EXISTS.
        """
        await self._create_constraints()
        await self._create_vector_indexes()
        await self._create_fulltext_indexes()
        await self._create_composite_indexes()

    async def _create_constraints(self) -> None:
        """
        Uniqueness constraints — prevent duplicate nodes.
        
        Each node type has a unique ID field. If the extraction pipeline
        tries to create a node with an ID that already exists, Neo4j
        will reject it (or MERGE will find the existing one).
        
        This is our primary defense against data duplication.
        """
        constraints = [
            # Content Layer
            "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.episodeId IS UNIQUE",
            "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segmentId IS UNIQUE",
            "CREATE CONSTRAINT utterance_id IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utteranceId IS UNIQUE",
            # Entity Layer
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.personId IS UNIQUE",
            "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.orgId IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topicId IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (ev:Event) REQUIRE ev.eventId IS UNIQUE",
            # Analytical Layer
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claimId IS UNIQUE",
            # Community Layer
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (com:Community) REQUIRE com.communityId IS UNIQUE",
        ]

        for constraint in constraints:
            await self.run(constraint)

    async def _create_vector_indexes(self) -> None:
        """
        Vector indexes — enable fast semantic similarity search.
        
        These use Neo4j's native vector index (ANN — Approximate Nearest Neighbor).
        Each index is configured for:
        - 1536 dimensions (text-embedding-3-small from OpenAI)
        - Cosine similarity (standard for text embeddings)
        
        We create vector indexes on:
        - Utterance (chunk) embeddings — for finding relevant passages
        - Claim embeddings — for finding similar claims
        - Entity embeddings (Person, Organization, Topic, Event) — for entity search
        - Community embeddings — for thematic search
        """
        dim = settings.embedding_dimensions  # 1536

        vector_indexes = [
            # Content layer — find relevant chunks by semantic similarity
            f"""CREATE VECTOR INDEX utterance_embeddings IF NOT EXISTS
                FOR (u:Utterance) ON (u.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            # Analytical layer — find similar claims
            f"""CREATE VECTOR INDEX claim_embeddings IF NOT EXISTS
                FOR (c:Claim) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            # Entity layer — find entities by semantic description
            f"""CREATE VECTOR INDEX person_embeddings IF NOT EXISTS
                FOR (p:Person) ON (p.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            f"""CREATE VECTOR INDEX org_embeddings IF NOT EXISTS
                FOR (o:Organization) ON (o.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            f"""CREATE VECTOR INDEX topic_embeddings IF NOT EXISTS
                FOR (t:Topic) ON (t.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            f"""CREATE VECTOR INDEX event_embeddings IF NOT EXISTS
                FOR (ev:Event) ON (ev.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",

            # Community layer — find relevant thematic clusters
            f"""CREATE VECTOR INDEX community_embeddings IF NOT EXISTS
                FOR (com:Community) ON (com.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
        ]

        for index in vector_indexes:
            await self.run(index)

    async def _create_fulltext_indexes(self) -> None:
        """
        Full-text indexes — enable keyword search.
        
        These support queries like "find all chunks mentioning 'Alexander'"
        without needing embeddings. Useful as a complement to vector search
        (hybrid retrieval = vector + keyword).
        
        Full-text indexes use Lucene under the hood and support:
        - Tokenization (splits text into searchable words)
        - Stemming (optional)
        - Fuzzy matching
        """
        fulltext_indexes = [
            # Search within chunk text
            "CREATE FULLTEXT INDEX utterance_text IF NOT EXISTS FOR (u:Utterance) ON EACH [u.text]",
            # Search within claim text
            "CREATE FULLTEXT INDEX claim_text IF NOT EXISTS FOR (c:Claim) ON EACH [c.text]",
        ]

        for index in fulltext_indexes:
            await self.run(index)

    async def _create_composite_indexes(self) -> None:
        """
        Composite indexes — speed up common query patterns.
        
        These are traditional B-tree indexes on properties we frequently
        filter by. Without them, Neo4j would scan all nodes of that label.
        
        We index:
        - Episode.publishDate — for temporal queries ("claims from 2024")
        - Claim.claimType — for filtering by type ("all predictions")
        - Claim.status — for filtering by status ("verified predictions")
        - Claim.stance — for filtering by stance ("critical claims")
        - Topic.category — for filtering topics by category
        """
        composite_indexes = [
            "CREATE INDEX episode_date IF NOT EXISTS FOR (e:Episode) ON (e.publishDate)",
            "CREATE INDEX claim_type IF NOT EXISTS FOR (c:Claim) ON (c.claimType)",
            "CREATE INDEX claim_status IF NOT EXISTS FOR (c:Claim) ON (c.status)",
            "CREATE INDEX claim_stance IF NOT EXISTS FOR (c:Claim) ON (c.stance)",
            "CREATE INDEX topic_category IF NOT EXISTS FOR (t:Topic) ON (t.category)",
        ]

        for index in composite_indexes:
            await self.run(index)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def get_node_counts(self) -> dict[str, int]:
        """
        Get the count of each node type in the graph.
        Useful for monitoring and validation after ingestion.
        """
        result = await self.run("""
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {}) YIELD value
            RETURN label, value.count AS count
        """)
        return {r["label"]: r["count"] for r in result}

    async def get_relationship_counts(self) -> dict[str, int]:
        """
        Get the count of each relationship type in the graph.
        Useful for monitoring and validation after ingestion.
        """
        result = await self.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            CALL apoc.cypher.run(
                'MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) AS count', {}
            ) YIELD value
            RETURN relationshipType, value.count AS count
        """)
        return {r["relationshipType"]: r["count"] for r in result}

    async def clear_database(self) -> None:
        """
        ⚠️ DESTRUCTIVE: Delete ALL nodes and relationships.
        Use only for testing or starting fresh.
        """
        await self.run("MATCH (n) DETACH DELETE n")
