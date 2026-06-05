"""
Centralized configuration for the entire project.
=================================================

All settings live here. Every other module imports from this file
instead of hardcoding values or reading .env directly.

HOW IT WORKS:
    We use Pydantic's BaseSettings which automatically reads from:
    1. Environment variables (highest priority)
    2. A .env file in the project root (fallback)
    3. Default values defined here (lowest priority)

    So if you set OPENAI_API_KEY in your .env file, it gets picked up
    automatically. If you set it as an actual environment variable,
    that takes priority over .env.

WHY CENTRALIZED:
    - One place to see ALL configurable values
    - Easy to change parameters without hunting through code
    - Type-safe: Pydantic validates that values make sense
    - Documented: each field has a description

USAGE IN OTHER MODULES:
    from src.config import settings
    
    client = OpenAI(api_key=settings.openai_api_key)
    chunks = chunk_transcript(text, max_chunk_tokens=settings.max_chunk_tokens)
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    All project settings. Loaded from environment variables / .env file.
    """

    # =========================================================================
    # API Keys
    # =========================================================================

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Used for all LLM calls and embeddings."
    )

    groq_api_key: str = Field(
        default="",
        description="Groq API key. Used for transcription (Whisper) in the MVP."
    )

    # =========================================================================
    # Neo4j Connection
    # =========================================================================

    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt connection URI."
    )

    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username."
    )

    neo4j_password: str = Field(
        default="",
        description="Neo4j password."
    )

    # =========================================================================
    # LLM Models
    # =========================================================================
    # We use OpenAI exclusively. Two tiers:
    # - "cheap" (gpt-4o-mini): high volume, structured extraction (Pass 1, 2, 5)
    # - "strong" (gpt-4o): nuanced reasoning (Pass 3, 4)
    # =========================================================================

    model_cheap: str = Field(
        default="gpt-4o-mini",
        description="Cost-efficient model for high-volume extraction (entities, relationships, entity resolution)."
    )

    model_strong: str = Field(
        default="gpt-5-mini",
        description="Capable model for nuanced tasks (claims, coreference resolution)."
    )

    model_segmentation: str = Field(
        default="gpt-4o-mini",
        description="Model used for Level 1 chunking (topic boundary detection). "
        "gpt-4o-mini is sufficient since segmentation is simpler than extraction."
    )

    # =========================================================================
    # Embedding Model
    # =========================================================================

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model. 1536 dimensions, cosine similarity."
    )

    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality of the embedding vectors."
    )

    # =========================================================================
    # Chunking Parameters
    # =========================================================================

    max_chunk_tokens: int = Field(
        default=1500,
        description="Maximum tokens per Level 2 chunk. "
        "Balances extraction quality (more context) vs. cost (fewer tokens per call)."
    )

    min_chunk_tokens: int = Field(
        default=200,
        description="Minimum tokens for a chunk to stand alone. "
        "Smaller chunks get merged into the previous one."
    )

    overlap_sentences: int = Field(
        default=3,
        description="Number of sentences repeated between consecutive chunks. "
        "Ensures cross-boundary assertions are captured."
    )

    segmentation_window_size: int = Field(
        default=20,
        description="How many sentences per LLM call in Level 1 segmentation."
    )

    segmentation_window_overlap: int = Field(
        default=3,
        description="Overlap between consecutive segmentation windows."
    )

    # =========================================================================
    # Extraction Parameters
    # =========================================================================

    extraction_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for an extraction to be ingested. "
        "Below this, the extraction is flagged for review."
    )

    coreference_window_chunks: int = Field(
        default=2,
        description="How many chunks to include in the sliding window for coreference resolution."
    )

    # =========================================================================
    # Entity Resolution Parameters
    # =========================================================================

    fuzzy_match_threshold: int = Field(
        default=85,
        description="RapidFuzz token_sort_ratio threshold for fuzzy entity matching. "
        "Pairs scoring above this are considered potential duplicates."
    )

    embedding_similarity_threshold: float = Field(
        default=0.90,
        description="Cosine similarity threshold for embedding-based entity matching."
    )

    # =========================================================================
    # Pydantic Settings Configuration
    # =========================================================================

    class Config:
        # Read from .env file in the project root
        env_file = ".env"
        # Environment variables are case-insensitive
        # So OPENAI_API_KEY, openai_api_key, and OpenAI_API_Key all work
        case_sensitive = False


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
# Import this in other modules:
#     from src.config import settings
#
# It's created once when first imported, then reused everywhere.
# =============================================================================

settings = Settings()
