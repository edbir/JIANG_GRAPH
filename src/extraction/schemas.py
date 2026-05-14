"""
Pydantic schemas for the multi-pass extraction pipeline.

Each pass produces structured output conforming to these models.
OpenAI Structured Outputs uses these schemas to guarantee 100% JSON compliance.

Pass 1: Entity Extraction (GPT-4o-mini)
Pass 2: Relationship Extraction (GPT-4o-mini)
Pass 3: Claim/Opinion Extraction (GPT-4o)
Pass 4: Coreference Resolution (GPT-4o)
Pass 5: Entity Resolution — LLM judgment stage (GPT-4o-mini)
"""

from enum import Enum
from pydantic import BaseModel, Field


# =============================================================================
# Enums — Constrained values for schema-guided extraction
# =============================================================================


class EntityType(str, Enum):
    """The four primary entity types in our ontology."""
    PERSON = "person"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    EVENT = "event"


class RelationshipType(str, Enum):
    """
    Allowed relationship types between entities.
    Schema-constrained: the LLM cannot invent new types.
    """
    ROLE_IN = "ROLE_IN"                    # Person → Organization
    PARTICIPATED_IN = "PARTICIPATED_IN"    # Person/Org → Event
    LED_TO = "LED_TO"                      # Event → Event (causal chain)
    CAUSED = "CAUSED"                      # Event → Event (direct cause)
    CONTROLLED = "CONTROLLED"              # Organization → Topic (territory)
    LOCATED_IN = "LOCATED_IN"              # Organization → Topic (geography)


class ClaimType(str, Enum):
    """Types of claims Prof. Jiang makes in his lectures."""
    FACTUAL_ASSERTION = "factual_assertion"
    HISTORICAL_INTERPRETATION = "historical_interpretation"
    GEOPOLITICAL_PREDICTION = "geopolitical_prediction"
    CAUSAL_ANALYSIS = "causal_analysis"


class Stance(str, Enum):
    """
    The speaker's stance on the claim.
    Replaces generic 'sentiment' with domain-appropriate labels.
    """
    CRITICAL = "critical"
    FAVORABLE = "favorable"
    NEUTRAL = "neutral"
    SPECULATIVE = "speculative"


# =============================================================================
# Pass 1 — Entity Extraction
# =============================================================================


class Entity(BaseModel):
    """A single entity extracted from a transcript chunk."""

    name: str = Field(
        description="Canonical, universally recognized name. "
        "E.g., 'Alexander the Great', 'Roman Empire', 'Battle of Thermopylae'."
    )
    entity_type: EntityType = Field(
        description="The type of entity."
    )
    description: str = Field(
        description="Brief description of the entity as mentioned in this chunk's context."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names, titles, or abbreviations mentioned in the text. "
        "E.g., ['Alexander', 'Alexander III of Macedon']."
    )
    historical_role: str | None = Field(
        default=None,
        description="Only for entity_type=person. "
        "One of: ruler, military_leader, philosopher, diplomat, civilian, "
        "religious_leader, scientist, artist."
    )
    org_type: str | None = Field(
        default=None,
        description="Only for entity_type=organization. "
        "One of: empire, nation-state, dynasty, alliance, religion, institution."
    )
    topic_category: str | None = Field(
        default=None,
        description="Only for entity_type=topic. "
        "One of: geography, ideology, economic_system, governing_system, "
        "military_strategy, religious_practice."
    )
    event_type: str | None = Field(
        default=None,
        description="Only for entity_type=event. "
        "One of: war, battle, treaty, revolution, crisis, period."
    )


class EntityExtractionResult(BaseModel):
    """Output of Pass 1: all entities found in a single chunk."""
    entities: list[Entity] = Field(
        description="List of entities extracted from the transcript chunk."
    )


# =============================================================================
# Pass 2 — Relationship Extraction
# =============================================================================


class Relationship(BaseModel):
    """A directed relationship between two entities."""

    source_entity: str = Field(
        description="Name of the source entity (must match a name from Pass 1)."
    )
    target_entity: str = Field(
        description="Name of the target entity (must match a name from Pass 1)."
    )
    relationship_type: RelationshipType = Field(
        description="The type of relationship, constrained to the graph schema."
    )
    description: str = Field(
        description="Natural language description of the relationship. "
        "E.g., 'Julius Caesar served as dictator of the Roman Republic'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Extraction confidence. Below 0.7 will be flagged for review."
    )


class RelationshipExtractionResult(BaseModel):
    """Output of Pass 2: all relationships found in a single chunk."""
    relationships: list[Relationship] = Field(
        description="List of relationships between entities in this chunk."
    )


# =============================================================================
# Pass 3 — Claim/Opinion Extraction
# =============================================================================


class Claim(BaseModel):
    """
    A reified claim, interpretation, prediction, or causal assertion
    made by Prof. Jiang. Claims are first-class nodes in the graph.
    """

    text: str = Field(
        description="The claim as a clear, standalone statement. "
        "E.g., 'The fall of the Roman Republic was primarily driven by economic inequality'."
    )
    claim_type: ClaimType = Field(
        description="The type of claim."
    )
    stance: Stance = Field(
        description="Prof. Jiang's stance when making this claim."
    )
    entities_involved: list[str] = Field(
        description="Names of entities referenced by this claim. "
        "Must correspond to entities extracted in Pass 1."
    )
    time_reference: str | None = Field(
        default=None,
        description="Historical era, date, or future timeline the claim references. "
        "E.g., '3rd century BC', '2025', 'Bronze Age collapse'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Extraction confidence. Below 0.7 will be flagged for review."
    )
    # Prediction-specific fields (only when claim_type == geopolitical_prediction)
    target_date: str | None = Field(
        default=None,
        description="Only for geopolitical_prediction. "
        "The date by which the prediction should resolve. E.g., '2026', 'within 5 years'."
    )
    target_description: str | None = Field(
        default=None,
        description="Only for geopolitical_prediction. "
        "What is expected to happen. E.g., 'The US will experience a civil conflict'."
    )


class ClaimExtractionResult(BaseModel):
    """Output of Pass 3: all claims found in a single chunk."""
    claims: list[Claim] = Field(
        description="List of claims, opinions, and predictions extracted from the chunk."
    )


# =============================================================================
# Pass 4 — Coreference Resolution
# =============================================================================


class CoreferenceEntry(BaseModel):
    """A single resolved coreference: an ambiguous reference mapped to a canonical entity."""

    reference_text: str = Field(
        description="The ambiguous text in the chunk. "
        "E.g., 'him', 'the empire', 'that city-state', 'the philosopher'."
    )
    resolved_entity: str = Field(
        description="The canonical entity name this reference resolves to. "
        "E.g., 'Alexander the Great', 'Persian Empire', 'Athens'."
    )
    chunk_index: int = Field(
        description="Index of the chunk (within the sliding window) where this reference appears."
    )


class CoreferenceResult(BaseModel):
    """Output of Pass 4: all coreferences resolved in a sliding window of chunks."""
    coreferences: list[CoreferenceEntry] = Field(
        description="List of resolved coreferences across the chunk window."
    )


# =============================================================================
# Pass 5 — Entity Resolution (LLM judgment for ambiguous pairs)
# =============================================================================


class EntityResolutionJudgment(BaseModel):
    """
    LLM decision on whether two entities should be merged.
    Only called for the ~5% of pairs that the cheaper methods
    (exact match, fuzzy, embedding) could not resolve.
    """

    entity_a: str = Field(
        description="Name of the first entity."
    )
    entity_b: str = Field(
        description="Name of the second entity."
    )
    should_merge: bool = Field(
        description="Whether these two entities refer to the same real-world entity."
    )
    canonical_name: str = Field(
        description="If should_merge=True, the canonical name to use. "
        "If should_merge=False, repeat entity_a."
    )
    reasoning: str = Field(
        description="Brief justification for the decision. Enables human audit."
    )


class EntityResolutionBatch(BaseModel):
    """Output of Pass 5: resolution decisions for a batch of ambiguous entity pairs."""
    judgments: list[EntityResolutionJudgment] = Field(
        description="List of merge/no-merge decisions for entity pairs."
    )
