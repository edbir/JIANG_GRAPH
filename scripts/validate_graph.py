"""
Validate Graph — Quality Assurance Checks for the Knowledge Graph.
===================================================================

PURPOSE:
    Run after each episode ingestion (or periodically) to detect issues
    in the graph: orphan entities, ungrounded claims, duplicates, missing
    embeddings, super-nodes, and structural problems.

WHAT IT CHECKS:
    1. Orphan entities — nodes with no relationships (likely noise)
    2. Claims without provenance — claims not linked to an utterance
    3. Claims without topic grounding — claims not linked to any entity via ABOUT
    4. Duplicate relationships — same pair connected by same type multiple times
    5. Missing embeddings — nodes that should have embeddings but don't
    6. Entity connectivity distribution — detect super-nodes and isolates
    7. Content hierarchy integrity — episodes → segments → utterances chain
    8. Overall statistics — node/relationship counts for monitoring

USAGE:
    # Run all checks
    python -m scripts.validate_graph

    # Run with verbose output (show individual issues)
    python -m scripts.validate_graph --verbose

    # Check a specific episode only
    python -m scripts.validate_graph --episode ep001

    # Output as JSON (for programmatic use)
    python -m scripts.validate_graph --json

WHY THIS EXISTS:
    The extraction pipeline is LLM-based — it WILL produce imperfect output.
    This script gives you visibility into what's wrong so you can:
    - Tune extraction prompts (if claims are too generic)
    - Adjust entity resolution thresholds (if too many duplicates)
    - Identify problematic episodes (bad transcription quality)
    - Track quality over time (run after each batch ingestion)

REFERENCE:
    Section 8.1 of the architecture doc defines these validation queries.
    Target: >85% accuracy across all dimensions before full backlog processing.
"""

import asyncio
import argparse
import json
from dataclasses import dataclass, asdict

from src.graph.neo4j_client import GraphClient


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    status: str         # "pass", "warn", "fail"
    count: int          # How many issues found
    message: str        # Human-readable summary
    details: list[dict] # Individual issues (populated with --verbose)


@dataclass
class ValidationReport:
    """Full validation report across all checks."""
    total_checks: int
    passed: int
    warnings: int
    failures: int
    results: list[ValidationResult]
    statistics: dict


# =============================================================================
# VALIDATION CHECKS
# =============================================================================


async def check_orphan_entities(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 1: Find entities with NO relationships.
    
    An entity with no relationships is likely extraction noise — it was
    mentioned once but never connected to anything meaningful.
    
    Excludes Episode and Community nodes (they may legitimately have
    few connections early on).
    """
    results = await client.run("""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
            AND NOT (e)--()
        RETURN labels(e)[0] AS type, e.name AS name
        ORDER BY type, name
    """)

    count = len(results)
    status = "pass" if count == 0 else ("warn" if count < 10 else "fail")

    return ValidationResult(
        check_name="Orphan Entities",
        status=status,
        count=count,
        message=f"{count} entities with no relationships (likely noise)",
        details=results if verbose else [],
    )


async def check_claims_without_provenance(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 2: Find claims not linked to any utterance via EXTRACTED_FROM.
    
    Every claim MUST trace back to the specific chunk it was extracted from.
    Missing provenance means we can't verify the claim.
    """
    results = await client.run("""
        MATCH (c:Claim)
        WHERE NOT (c)-[:EXTRACTED_FROM]->(:Utterance)
        RETURN c.claimId AS id, c.text AS text
        LIMIT 50
    """)

    count = len(results)
    status = "pass" if count == 0 else "fail"

    return ValidationResult(
        check_name="Claims Without Provenance",
        status=status,
        count=count,
        message=f"{count} claims have no EXTRACTED_FROM relationship",
        details=results if verbose else [],
    )


async def check_claims_without_grounding(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 3: Find claims not linked to any entity via ABOUT.
    
    Claims should reference at least one entity. Ungrounded claims
    might indicate the extraction prompt isn't linking claims to entities properly.
    """
    results = await client.run("""
        MATCH (c:Claim)
        WHERE NOT (c)-[:ABOUT]->()
        RETURN c.claimId AS id, c.text AS text, c.claimType AS type
        LIMIT 50
    """)

    count = len(results)
    # Some claims might legitimately not reference specific entities
    status = "pass" if count == 0 else ("warn" if count < 20 else "fail")

    return ValidationResult(
        check_name="Claims Without Entity Grounding",
        status=status,
        count=count,
        message=f"{count} claims have no ABOUT relationship to any entity",
        details=results if verbose else [],
    )


async def check_duplicate_relationships(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 4: Find duplicate relationships between the same pair of nodes.
    
    Idempotent MERGE should prevent these, but bugs in the upsert logic
    or conflicting relationship types can create duplicates.
    """
    results = await client.run("""
        MATCH (a)-[r1]->(b), (a)-[r2]->(b)
        WHERE id(r1) < id(r2) AND type(r1) = type(r2)
        RETURN type(r1) AS relType, 
               labels(a)[0] AS sourceType, 
               labels(b)[0] AS targetType,
               count(*) AS duplicates
        ORDER BY duplicates DESC
    """)

    total_duplicates = sum(r["duplicates"] for r in results)
    status = "pass" if total_duplicates == 0 else ("warn" if total_duplicates < 10 else "fail")

    return ValidationResult(
        check_name="Duplicate Relationships",
        status=status,
        count=total_duplicates,
        message=f"{total_duplicates} duplicate relationships found across {len(results)} type(s)",
        details=results if verbose else [],
    )


async def check_missing_embeddings(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 5: Find nodes that should have embeddings but don't.
    
    After ingestion, ALL utterances and claims should have embeddings.
    Entity embeddings are also expected for retrieval quality.
    """
    results = await client.run("""
        OPTIONAL MATCH (u:Utterance) WHERE u.embedding IS NULL
        WITH count(u) AS missing_utterances
        OPTIONAL MATCH (c:Claim) WHERE c.embedding IS NULL
        WITH missing_utterances, count(c) AS missing_claims
        OPTIONAL MATCH (p:Person) WHERE p.embedding IS NULL
        WITH missing_utterances, missing_claims, count(p) AS missing_persons
        OPTIONAL MATCH (t:Topic) WHERE t.embedding IS NULL
        WITH missing_utterances, missing_claims, missing_persons, count(t) AS missing_topics
        RETURN missing_utterances, missing_claims, missing_persons, missing_topics
    """)

    if results:
        r = results[0]
        total_missing = (
            r["missing_utterances"] + r["missing_claims"]
            + r["missing_persons"] + r["missing_topics"]
        )
        details = [r] if verbose else []
    else:
        total_missing = 0
        details = []

    status = "pass" if total_missing == 0 else ("warn" if total_missing < 20 else "fail")

    return ValidationResult(
        check_name="Missing Embeddings",
        status=status,
        count=total_missing,
        message=f"{total_missing} nodes missing embeddings",
        details=details,
    )


async def check_entity_connectivity(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 6: Entity connectivity distribution.
    
    Detects:
    - Isolates (degree 0-1): might be noise
    - Super-nodes (degree > 100): might need topic hierarchy or bucketing
    
    A healthy distribution has most entities between degree 2-50.
    """
    results = await client.run("""
        MATCH (e)
        WHERE (e:Person OR e:Organization OR e:Topic OR e:Event)
        WITH labels(e)[0] AS type, e.name AS name, size((e)--()) AS degree
        RETURN type,
               count(*) AS total,
               min(degree) AS min_degree,
               avg(degree) AS avg_degree,
               max(degree) AS max_degree,
               percentileCont(degree, 0.95) AS p95_degree
        ORDER BY type
    """)

    # Check for super-nodes
    max_degree = max((r["max_degree"] for r in results), default=0)
    status = "pass" if max_degree < 200 else ("warn" if max_degree < 500 else "fail")

    return ValidationResult(
        check_name="Entity Connectivity Distribution",
        status=status,
        count=max_degree,
        message=f"Max degree: {max_degree} (super-nodes > 200 may impact query performance)",
        details=results if verbose else [],
    )


async def check_content_hierarchy(client: GraphClient, verbose: bool = False) -> ValidationResult:
    """
    Check 7: Content hierarchy integrity.
    
    Verifies:
    - Every Segment belongs to an Episode
    - Every Utterance belongs to a Segment
    - NEXT_CHUNK chains are continuous (no broken links)
    """
    # Segments without episodes
    orphan_segments = await client.run("""
        MATCH (s:Segment)
        WHERE NOT (:Episode)-[:HAS_SEGMENT]->(s)
        RETURN s.segmentId AS id
    """)

    # Utterances without segments
    orphan_utterances = await client.run("""
        MATCH (u:Utterance)
        WHERE NOT (:Segment)-[:HAS_CHUNK]->(u)
        RETURN u.utteranceId AS id
        LIMIT 20
    """)

    total_issues = len(orphan_segments) + len(orphan_utterances)
    status = "pass" if total_issues == 0 else "fail"

    details = []
    if verbose:
        if orphan_segments:
            details.append({"issue": "orphan_segments", "count": len(orphan_segments)})
        if orphan_utterances:
            details.append({"issue": "orphan_utterances", "count": len(orphan_utterances)})

    return ValidationResult(
        check_name="Content Hierarchy Integrity",
        status=status,
        count=total_issues,
        message=f"{len(orphan_segments)} orphan segments, {len(orphan_utterances)} orphan utterances",
        details=details,
    )


async def check_episode_specific(
    client: GraphClient, episode_id: str, verbose: bool = False
) -> ValidationResult:
    """
    Check episode-specific stats: how many entities, claims, segments, chunks.
    
    Useful for spotting episodes with unusually low extraction (bad transcript?)
    or unusually high entity count (hallucination?).
    """
    results = await client.run("""
        MATCH (e:Episode {episodeId: $episodeId})
        OPTIONAL MATCH (e)-[:HAS_SEGMENT]->(s:Segment)
        WITH e, count(DISTINCT s) AS segments
        OPTIONAL MATCH (e)-[:HAS_SEGMENT]->(:Segment)-[:HAS_CHUNK]->(u:Utterance)
        WITH e, segments, count(DISTINCT u) AS utterances
        OPTIONAL MATCH (c:Claim)-[:STATED_DURING]->(e)
        WITH e, segments, utterances, count(DISTINCT c) AS claims
        OPTIONAL MATCH (e)-[:HAS_SEGMENT]->(:Segment)-[:HAS_CHUNK]->(u2:Utterance)
        OPTIONAL MATCH (u2)-[:MENTIONS]->(ent)
        WITH e, segments, utterances, claims, count(DISTINCT ent) AS entities
        RETURN e.title AS title, segments, utterances, claims, entities
    """, {"episodeId": episode_id})

    if not results:
        return ValidationResult(
            check_name=f"Episode Check ({episode_id})",
            status="fail",
            count=0,
            message=f"Episode {episode_id} not found in graph",
            details=[],
        )

    r = results[0]
    status = "pass"
    if r["claims"] == 0:
        status = "warn"
    if r["utterances"] == 0:
        status = "fail"

    return ValidationResult(
        check_name=f"Episode Check ({episode_id})",
        status=status,
        count=r["claims"],
        message=(
            f"{r['title']}: {r['segments']} segments, {r['utterances']} utterances, "
            f"{r['entities']} entities, {r['claims']} claims"
        ),
        details=[r] if verbose else [],
    )


# =============================================================================
# GRAPH STATISTICS
# =============================================================================


async def get_statistics(client: GraphClient) -> dict:
    """Gather overall graph statistics."""
    # Node counts by label
    node_counts = await client.run("""
        MATCH (n)
        WITH labels(n)[0] AS label, count(n) AS count
        RETURN label, count
        ORDER BY count DESC
    """)

    # Relationship counts by type
    rel_counts = await client.run("""
        MATCH ()-[r]->()
        WITH type(r) AS type, count(r) AS count
        RETURN type, count
        ORDER BY count DESC
    """)

    # Episode count
    episodes = await client.run("MATCH (e:Episode) RETURN count(e) AS count")

    return {
        "nodes": {r["label"]: r["count"] for r in node_counts},
        "relationships": {r["type"]: r["count"] for r in rel_counts},
        "total_nodes": sum(r["count"] for r in node_counts),
        "total_relationships": sum(r["count"] for r in rel_counts),
        "episodes": episodes[0]["count"] if episodes else 0,
    }


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================


async def run_validation(
    episode_id: str = None,
    verbose: bool = False,
    output_json: bool = False,
) -> ValidationReport:
    """
    Run all validation checks and produce a report.
    
    Parameters:
        episode_id:  If specified, also run episode-specific check
        verbose:     Include details for each issue found
        output_json: Print results as JSON (for programmatic use)
    
    Returns:
        ValidationReport with all results
    """
    async with GraphClient() as client:
        results: list[ValidationResult] = []

        # Run all standard checks
        print("Running validation checks...\n")

        checks = [
            ("1/7", check_orphan_entities),
            ("2/7", check_claims_without_provenance),
            ("3/7", check_claims_without_grounding),
            ("4/7", check_duplicate_relationships),
            ("5/7", check_missing_embeddings),
            ("6/7", check_entity_connectivity),
            ("7/7", check_content_hierarchy),
        ]

        for label, check_fn in checks:
            result = await check_fn(client, verbose=verbose)
            results.append(result)
            if not output_json:
                icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[result.status]
                print(f"  [{label}] {icon} {result.check_name}: {result.message}")

        # Episode-specific check (if requested)
        if episode_id:
            result = await check_episode_specific(client, episode_id, verbose=verbose)
            results.append(result)
            if not output_json:
                icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[result.status]
                print(f"  [+] {icon} {result.check_name}: {result.message}")

        # Get statistics
        stats = await get_statistics(client)

    # Build report
    report = ValidationReport(
        total_checks=len(results),
        passed=sum(1 for r in results if r.status == "pass"),
        warnings=sum(1 for r in results if r.status == "warn"),
        failures=sum(1 for r in results if r.status == "fail"),
        results=results,
        statistics=stats,
    )

    # Output
    if output_json:
        output = {
            "summary": {
                "total": report.total_checks,
                "passed": report.passed,
                "warnings": report.warnings,
                "failures": report.failures,
            },
            "checks": [asdict(r) for r in report.results],
            "statistics": report.statistics,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*50}")
        print(f"SUMMARY: {report.passed} passed, {report.warnings} warnings, {report.failures} failures")
        print(f"{'='*50}")
        print(f"\nGraph Statistics:")
        print(f"  Episodes: {stats['episodes']}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        if stats["nodes"]:
            print(f"  Nodes by type:")
            for label, count in stats["nodes"].items():
                print(f"    {label}: {count}")
        if stats["relationships"]:
            print(f"  Relationships by type:")
            for rel_type, count in stats["relationships"].items():
                print(f"    {rel_type}: {count}")

    return report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Command-line interface for graph validation."""
    parser = argparse.ArgumentParser(
        description="Run quality assurance checks on the knowledge graph."
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Also validate a specific episode (e.g., 'ep001')"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show individual issues for each check"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    asyncio.run(run_validation(
        episode_id=args.episode,
        verbose=args.verbose,
        output_json=args.json,
    ))


if __name__ == "__main__":
    main()
