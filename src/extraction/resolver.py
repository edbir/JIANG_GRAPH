"""
Entity Resolution Cascade.
===========================

PURPOSE:
    After extraction, the same real-world entity appears with different names
    across chunks and episodes. This module deduplicates them using a cascade
    that goes from cheap/fast methods to expensive/accurate ones.

THE CASCADE (in order):
    Stage 1: Exact match after normalization     (~40% of duplicates) — FREE
    Stage 2: Fuzzy string matching (RapidFuzz)   (~35% of duplicates) — FREE
    Stage 3: Embedding similarity                (~20% of duplicates) — CHEAP (API)
    Stage 4: LLM judgment                        (~5% of duplicates)  — MODERATE (API)

DESIGN PRINCIPLE:
    "When in doubt, do NOT merge."
    A false positive (incorrect merge) corrupts the graph and is hard to undo.
    A false negative (keeping duplicates) is harmless and fixable later.

WHAT THIS MODULE DOES NOT DO:
    - Does NOT touch Neo4j (that's graph/upsert.py's job)
    - Does NOT run the extraction passes (that's extractor.py's job)
    - Does NOT handle coreference ("he" → "Alexander") — that's Pass 4

WHAT IT PRODUCES:
    A merge map: dict[str, str] mapping each entity name to its canonical form.
    Example: {"Alexander": "Alexander the Great", "Alexander III": "Alexander the Great"}
"""

import unicodedata
import re
from dataclasses import dataclass

from src.config import settings


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EntityCandidate:
    """
    An entity candidate for resolution.
    
    We collect these from all EntityExtractionResult objects across all chunks,
    then compare them pairwise to find duplicates.
    """
    name: str                   # The extracted name
    entity_type: str            # person, organization, topic, event
    description: str            # Brief description from extraction context
    aliases: list[str]          # Known aliases from extraction
    source_chunks: list[str]    # Which chunks mentioned this entity


@dataclass
class MergeDecision:
    """
    A decision to merge two entities (or not).
    Stored for auditability — we can review all merge decisions later.
    """
    entity_a: str               # Name of first entity
    entity_b: str               # Name of second entity
    should_merge: bool          # The decision
    canonical_name: str         # Which name to keep (if merging)
    method: str                 # Which stage decided: "exact", "fuzzy", "embedding", "llm"
    score: float                # The similarity score that triggered the decision
    reasoning: str              # Why (especially useful for LLM decisions)


# =============================================================================
# STAGE 1 — EXACT MATCH (after normalization)
# =============================================================================
# This catches the obvious cases: same name with different casing,
# extra whitespace, or accent variations.
#
# Cost: Zero. Pure string operations.
# Catches: ~40% of all duplicates.
# =============================================================================


def normalize_name(name: str) -> str:
    """
    Normalize an entity name for exact comparison.
    
    Steps:
    1. Unicode NFC normalization (consistent character forms)
    2. Lowercase
    3. Strip leading/trailing whitespace
    4. Collapse multiple spaces into one
    5. Remove accents (é → e, ñ → n) for cross-accent matching
    
    Examples:
        "Alexander the Great" → "alexander the great"
        "  Roman   Empire  " → "roman empire"
        "São Paulo" → "sao paulo"
        "UNITED STATES" → "united states"
    """
    # NFC normalization
    text = unicodedata.normalize("NFC", name)

    # Lowercase
    text = text.lower()

    # Strip and collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove accents: decompose (NFD) then remove combining characters
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    return text


def find_exact_matches(candidates: list[EntityCandidate]) -> list[MergeDecision]:
    """
    Stage 1: Find entities that are identical after normalization.
    
    ALGORITHM:
    1. Group candidates by entity_type (never compare across types)
    2. Within each type, normalize all names
    3. If two candidates have the same normalized name → merge
    4. Choose the LONGEST original name as canonical (usually most complete)
    
    Also checks aliases: if entity A's name matches entity B's alias
    (after normalization), that's also a match.
    
    Returns:
        List of MergeDecision objects for all exact matches found
    """
    decisions: list[MergeDecision] = []

    # Group by type (person vs person, org vs org, etc.)
    by_type: dict[str, list[EntityCandidate]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate.entity_type, []).append(candidate)

    # Within each type, find exact matches
    for entity_type, type_candidates in by_type.items():
        # Build a map: normalized_name → list of candidates with that normalized name
        norm_groups: dict[str, list[EntityCandidate]] = {}

        for candidate in type_candidates:
            # Check the main name
            norm = normalize_name(candidate.name)
            norm_groups.setdefault(norm, []).append(candidate)

            # Also index by normalized aliases
            for alias in candidate.aliases:
                norm_alias = normalize_name(alias)
                norm_groups.setdefault(norm_alias, []).append(candidate)

        # For each group with >1 candidate, they're duplicates
        for norm_name, group in norm_groups.items():
            if len(group) <= 1:
                continue

            # Deduplicate the group (same candidate might appear via name AND alias)
            unique_names = list(set(c.name for c in group))
            if len(unique_names) <= 1:
                continue

            # Choose canonical: longest name (usually most complete/formal)
            canonical = max(unique_names, key=len)

            # Create merge decisions for all non-canonical names
            for name in unique_names:
                if name != canonical:
                    decisions.append(MergeDecision(
                        entity_a=name,
                        entity_b=canonical,
                        should_merge=True,
                        canonical_name=canonical,
                        method="exact",
                        score=1.0,
                        reasoning=f"Identical after normalization: '{normalize_name(name)}'",
                    ))

    return decisions


# =============================================================================
# STAGE 2 — FUZZY STRING MATCHING
# =============================================================================
# Uses RapidFuzz library for fast fuzzy comparison.
# token_sort_ratio handles word reordering:
#   "Philip the Second" vs "Second Philip the" → high score
#
# Cost: Negligible. RapidFuzz is C-optimized.
# Catches: ~35% of duplicates (after Stage 1 already removed the obvious ones).
# =============================================================================


def find_fuzzy_matches(
    candidates: list[EntityCandidate],
    already_merged: set[str],
    threshold: int = None,
) -> list[MergeDecision]:
    """
    Stage 2: Find entities with similar (but not identical) names.
    
    ALGORITHM:
    1. Group candidates by entity_type
    2. Skip any candidates already resolved in Stage 1
    3. Compare all pairs within each type using token_sort_ratio
    4. Pairs scoring above threshold are marked as duplicates
    
    WHY token_sort_ratio (not plain ratio):
        It sorts the tokens alphabetically before comparing.
        "Philip the Second" vs "the Second Philip" → 100% match
        "Battle of Thermopylae" vs "Thermopylae Battle" → high match
        
        This handles the common case where the same entity is described
        with words in different order.
    
    Parameters:
        candidates:     All entity candidates
        already_merged:  Set of names already resolved in Stage 1 (skip these)
        threshold:      Minimum score to consider a match (default from config: 85)
    
    Returns:
        List of MergeDecision objects for fuzzy matches
    """
    if threshold is None:
        threshold = settings.fuzzy_match_threshold

    # Import here so the module loads even without rapidfuzz installed
    try:
        from rapidfuzz import fuzz
    except ImportError:
        # If rapidfuzz isn't installed, skip this stage
        # (useful during early development)
        return []

    decisions: list[MergeDecision] = []

    # Group by type
    by_type: dict[str, list[EntityCandidate]] = {}
    for candidate in candidates:
        if candidate.name not in already_merged:
            by_type.setdefault(candidate.entity_type, []).append(candidate)

    # Compare all pairs within each type
    for entity_type, type_candidates in by_type.items():
        for i in range(len(type_candidates)):
            for j in range(i + 1, len(type_candidates)):
                a = type_candidates[i]
                b = type_candidates[j]

                # Compare using token_sort_ratio
                score = fuzz.token_sort_ratio(a.name, b.name)

                if score >= threshold:
                    # Choose canonical: longer name is usually more complete
                    canonical = a.name if len(a.name) >= len(b.name) else b.name
                    other = b.name if canonical == a.name else a.name

                    decisions.append(MergeDecision(
                        entity_a=other,
                        entity_b=canonical,
                        should_merge=True,
                        canonical_name=canonical,
                        method="fuzzy",
                        score=score / 100.0,  # Normalize to 0-1
                        reasoning=f"Fuzzy token_sort_ratio: {score}%",
                    ))

    return decisions


# =============================================================================
# STAGE 3 — EMBEDDING SIMILARITY
# =============================================================================
# Generates embeddings for "name + description" and compares by cosine similarity.
# Catches cases where names are very different but descriptions overlap.
#
# ⚠️ REQUIRES OPENAI API for embedding generation.
#
# Cost: One embedding call per entity (~$0.02 per 1M tokens — very cheap).
# Catches: ~20% of remaining duplicates.
# =============================================================================


async def find_embedding_matches(
    client,
    candidates: list[EntityCandidate],
    already_merged: set[str],
    threshold: float = None,
) -> list[MergeDecision]:
    """
    Stage 3: Find entities with similar embeddings (semantic similarity).
    
    ⚠️ REQUIRES OPENAI API KEY.
    
    ALGORITHM:
    1. For each unresolved candidate, create text: "name: description"
    2. Generate embeddings for all texts in one batch call
    3. Compare all pairs by cosine similarity
    4. Pairs above threshold (default 0.85) are marked as duplicates
    
    WHY "name + description" (not just name):
        - "USA" and "United States" have different name embeddings
        - But "USA: Federal republic in North America" and 
          "United States: Country in North America" have VERY similar embeddings
        - The description provides the semantic context that makes matching work
    
    Parameters:
        client:         OpenAI AsyncClient
        candidates:     All entity candidates
        already_merged: Set of names already resolved in Stages 1-2
        threshold:      Cosine similarity threshold (default from config: 0.85)
    
    Returns:
        List of MergeDecision objects for embedding matches
    """
    if threshold is None:
        threshold = settings.embedding_similarity_threshold

    import numpy as np

    decisions: list[MergeDecision] = []

    # Filter to unresolved candidates, grouped by type
    by_type: dict[str, list[EntityCandidate]] = {}
    for candidate in candidates:
        if candidate.name not in already_merged:
            by_type.setdefault(candidate.entity_type, []).append(candidate)

    for entity_type, type_candidates in by_type.items():
        if len(type_candidates) < 2:
            continue

        # Create embedding texts: "name: description"
        texts = [
            f"{c.name}: {c.description}" for c in type_candidates
        ]

        # Generate embeddings in batch
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )

        # Extract embedding vectors
        embeddings = np.array([item.embedding for item in response.data])

        # Compute cosine similarity matrix
        # Normalize vectors first (then dot product = cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = normalized @ normalized.T

        # Find pairs above threshold
        for i in range(len(type_candidates)):
            for j in range(i + 1, len(type_candidates)):
                score = float(similarity_matrix[i, j])

                if score >= threshold:
                    a = type_candidates[i]
                    b = type_candidates[j]

                    # Choose canonical: longer name
                    canonical = a.name if len(a.name) >= len(b.name) else b.name
                    other = b.name if canonical == a.name else a.name

                    decisions.append(MergeDecision(
                        entity_a=other,
                        entity_b=canonical,
                        should_merge=True,
                        canonical_name=canonical,
                        method="embedding",
                        score=score,
                        reasoning=f"Embedding cosine similarity: {score:.3f}",
                    ))

    return decisions


# =============================================================================
# STAGE 4 — LLM JUDGMENT
# =============================================================================
# Only for the ~5% of pairs that the cheaper methods couldn't resolve.
# Uses the extractor's resolve_entity_pair() function.
#
# ⚠️ REQUIRES OPENAI API.
# This stage is called from the main resolve() function below,
# but the actual LLM call lives in extractor.py (resolve_entity_pair).
# =============================================================================

# Stage 4 is handled by calling extractor.resolve_entity_pair() directly.
# We don't duplicate that logic here — the resolver orchestrates WHEN to call it,
# and the extractor handles HOW to call the LLM.


# =============================================================================
# MERGE MAP CONSTRUCTION
# =============================================================================


def build_merge_map(decisions: list[MergeDecision]) -> dict[str, str]:
    """
    Build the final merge map from all merge decisions.
    
    The merge map is a dict: {entity_name → canonical_name}
    
    TRANSITIVE CLOSURE:
        If A → B and B → C, then A should map to C (not B).
        We follow chains until we reach a name that maps to itself.
    
    Example:
        Decisions: "Alexander" → "Alexander the Great"
                   "Alexander III" → "Alexander the Great"
                   "The Macedonian" → "Alexander"
        
        Merge map: {"Alexander": "Alexander the Great",
                    "Alexander III": "Alexander the Great",
                    "The Macedonian": "Alexander the Great"}
        
        Note: "The Macedonian" → "Alexander" → "Alexander the Great" (transitive)
    
    Returns:
        Dict mapping every known name to its canonical form.
        Names that are already canonical map to themselves.
    """
    # Start with direct mappings from decisions
    raw_map: dict[str, str] = {}
    for decision in decisions:
        if decision.should_merge:
            raw_map[decision.entity_a] = decision.canonical_name

    # Resolve transitive chains
    # Keep following the map until we reach a fixed point
    def resolve(name: str) -> str:
        visited = set()
        current = name
        while current in raw_map and current not in visited:
            visited.add(current)
            current = raw_map[current]
        return current

    # Build final map with all transitives resolved
    merge_map: dict[str, str] = {}
    for name in raw_map:
        merge_map[name] = resolve(name)

    return merge_map


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def resolve_entities(
    candidates: list[EntityCandidate],
    client=None,
) -> tuple[dict[str, str], list[MergeDecision]]:
    """
    Run the full entity resolution cascade.
    
    This is the main function you call from outside.
    
    FLOW:
    1. Stage 1: Exact match (always runs, no dependencies)
    2. Stage 2: Fuzzy match (always runs, needs rapidfuzz)
    3. Stage 3: Embedding similarity (only if client provided)
    4. Build merge map from all decisions
    
    Stage 4 (LLM judgment) is NOT run automatically here.
    It should be called selectively for pairs that remain ambiguous
    after Stages 1-3, based on manual review or additional heuristics.
    
    Parameters:
        candidates: All EntityCandidate objects from all episodes
        client:     OpenAI AsyncClient (optional — needed for Stage 3)
    
    Returns:
        Tuple of:
        - merge_map: dict[str, str] mapping names to canonical forms
        - all_decisions: list[MergeDecision] for auditability
    """
    all_decisions: list[MergeDecision] = []
    merged_names: set[str] = set()

    # --- Stage 1: Exact match ---
    exact_decisions = find_exact_matches(candidates)
    all_decisions.extend(exact_decisions)
    merged_names.update(d.entity_a for d in exact_decisions if d.should_merge)

    # --- Stage 2: Fuzzy match ---
    fuzzy_decisions = find_fuzzy_matches(candidates, already_merged=merged_names)
    all_decisions.extend(fuzzy_decisions)
    merged_names.update(d.entity_a for d in fuzzy_decisions if d.should_merge)

    # --- Stage 3: Embedding similarity (only if API available) ---
    if client is not None:
        embedding_decisions = await find_embedding_matches(
            client=client,
            candidates=candidates,
            already_merged=merged_names,
        )
        all_decisions.extend(embedding_decisions)
        merged_names.update(d.entity_a for d in embedding_decisions if d.should_merge)

    # --- Build final merge map ---
    merge_map = build_merge_map(all_decisions)

    return merge_map, all_decisions
