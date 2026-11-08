"""Build the graph for one or more databases, start to finish.

    python scripts/build_graph.py --db adventureworks
    python scripts/build_graph.py --db adventureworks --rebuild
    python scripts/build_graph.py --all --skip-inference --skip-enrichment
    python scripts/build_graph.py --db adventureworks --skip-extraction     # reuse data/metadata/<db>.json

Per database, in order:

    extract + infer + save metadata   (extraction/build_metadata.py)   skipped by --skip-extraction
    ingest into Neo4j                 (graph/ingest.py)                --rebuild clears the db_id first
    apply edge weights                (graph/ingest.py)
    create vector indexes + embed     (graph/vector.py, graph/enrich.py)   skipped by --skip-enrichment
    verify the graph against the metadata

Generating missing descriptions with an LLM (`enrich.generate_missing_descriptions`) is not
written yet, so it is not a step here; embeddings use whatever descriptions the metadata has.

Everything that can fail for the whole run (Neo4j unreachable, no embedding config, missing API
key, a model whose vectors are not the configured size) is checked once, before any database is
touched. After that, one database failing does not stop the others; the exit code is 1 if any failed
or did not verify, 2 if the run could not start, and 3 if embedding was cut short by rate limiting.

Embedding is saved as it goes and resumes: after a rate-limited (partial) run, rerun the same command
without --rebuild and only the missing or changed nodes are embedded. --reembed redoes everything.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemagraph.models import DatabaseMetadata

log = logging.getLogger(__name__)

PROBE_TEXT = "dimension check"


class SetupError(RuntimeError):
    """The run cannot start; nothing was changed."""


@dataclass(frozen=True)
class Options:
    rebuild: bool = False
    skip_extraction: bool = False
    skip_inference: bool = False
    skip_enrichment: bool = False
    reembed: bool = False
    sample_n: int = 5
    min_overlap: float | None = None  # None: the inference default
    min_confidence: float = 0.5


@dataclass(frozen=True)
class Enrichment:
    """What the embedding steps need, resolved once for the whole run."""

    embedder: Any
    dimensions: int
    name: str


@dataclass
class Steps:
    """The operations a build performs. Tests replace them; `default_steps()` wires the real ones."""

    extract: Callable[..., Any]                      # (db_id, **options) -> build_metadata.BuildResult
    load_metadata: Callable[[Path], DatabaseMetadata]
    metadata_path: Callable[[str], Path]
    ingest: Callable[..., Any]                       # (meta, rebuild=) -> IngestReport
    apply_weights: Callable[[str], int]
    create_indexes: Callable[[int, str], None]       # (dimensions, name)
    embed: Callable[..., Any]                        # (db_id, embedder, name, reembed=) -> EmbedReport
    verify: Callable[[DatabaseMetadata], list[str]]


def default_steps() -> Steps:
    from schemagraph.extraction import build_metadata, extract
    from schemagraph.graph import enrich, ingest, vector

    def create_indexes(dimensions: int, name: str) -> None:
        vector.create_vector_indexes(dimensions, name=name)

    def do_embed(db_id: str, embedder: Any, name: str, reembed: bool = False) -> Any:
        return enrich.embed_nodes(db_id, embedder, name=name, reembed=reembed)

    return Steps(
        extract=build_metadata.build_one,
        load_metadata=extract.load_metadata,
        metadata_path=extract.default_metadata_path,
        ingest=ingest.ingest_metadata,
        apply_weights=ingest.apply_weights,
        create_indexes=create_indexes,
        embed=do_embed,
        verify=ingest.verify_against_metadata,
    )


@dataclass
class GraphResult:
    db_id: str
    steps_done: list[str] = field(default_factory=list)
    tables: int = 0
    columns: int = 0
    declared: int = 0
    inferred: int = 0
    edges_weighted: int = 0
    embedded_tables: int = 0       # embedded in this run
    embedded_columns: int = 0
    tables_current: int = 0        # with an up-to-date vector afterwards, of ...
    tables_total: int = 0
    columns_current: int = 0
    columns_total: int = 0
    partial: str | None = None     # embedding stopped early (rate limiting): why
    problems: list[str] = field(default_factory=list)  # verify differences
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.problems and self.partial is None

    @property
    def failed(self) -> bool:
        """A real failure, as opposed to a partial embedding run that can be resumed."""
        return self.error is not None or bool(self.problems)


# ---------------------------------------------------------------------------
# Whole-run checks
# ---------------------------------------------------------------------------


def prepare_enrichment(config: Any | None = None, make_embedder: Callable[[Any], Any] | None = None) -> Enrichment:
    """Resolve the embedder and prove it produces vectors of the configured size.

    A wrong size is the easy mistake to make: Neo4j accepts the vectors, but the index (created
    for the configured size) silently does not contain them, and retrieval returns nothing.
    One tiny embedding call here turns that into an error before any work is done.
    """
    from schemagraph.graph import vector

    if config is None:
        from schemagraph.config import get_config

        config = get_config().embedding
    if config is None:
        raise SetupError(
            "config.yml has no 'embedding' section (provider, model, dimensions, ...); "
            "add one (see config.example.yml) or use --skip-enrichment"
        )
    make = make_embedder or vector.make_embedder
    try:
        embedder = make(config)
        probe = embedder.embed_query(PROBE_TEXT)
    except Exception as exc:  # missing key, bad URL, HTTP error: all mean "cannot start"
        raise SetupError(f"cannot reach the embedding model ({config.provider}/{config.model}): {exc}") from exc
    if len(probe) != config.dimensions:
        raise SetupError(
            f"{config.provider}/{config.model} returned {len(probe)}-dimensional vectors but config.yml says "
            f"dimensions: {config.dimensions}. Fix the config (and set send_dimensions: true if you want "
            "shortened vectors); an index built with the wrong size would silently hold nothing."
        )
    vector.embedding_names(config.name)  # rejects a name that cannot go into Cypher
    return Enrichment(embedder=embedder, dimensions=config.dimensions, name=config.name)


# ---------------------------------------------------------------------------
# One database
# ---------------------------------------------------------------------------


def build_graph(
    db_id: str,
    options: Options | None = None,
    enrichment: Enrichment | None = None,
    steps: Steps | None = None,
) -> GraphResult:
    """Run every step for one database. Errors are captured in the result, not raised."""
    options = options or Options()
    steps = steps or default_steps()
    result = GraphResult(db_id)
    start = time.monotonic()
    try:
        path = steps.metadata_path(db_id)
        if options.skip_extraction:
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} does not exist; run without --skip-extraction first")
        else:
            built = steps.extract(
                db_id,
                skip_inference=options.skip_inference,
                sample_n=options.sample_n,
                **({"min_overlap": options.min_overlap} if options.min_overlap is not None else {}),
                min_confidence=options.min_confidence,
            )
            if not built.ok:
                raise RuntimeError(f"metadata extraction failed: {built.error}")
            result.steps_done.append("extract")

        meta = steps.load_metadata(path)
        if meta.db_id != db_id:
            raise ValueError(f"{path} is for db_id {meta.db_id!r}, not {db_id!r}")
        result.tables, result.columns = len(meta.tables), len(meta.columns)
        result.declared = sum(r.source == "declared" for r in meta.relationships)
        result.inferred = sum(r.source == "inferred" for r in meta.relationships)

        steps.ingest(meta, rebuild=options.rebuild)
        result.steps_done.append("ingest")
        result.edges_weighted = steps.apply_weights(db_id)
        result.steps_done.append("weights")

        if not options.skip_enrichment:
            if enrichment is None:
                raise RuntimeError("enrichment requested but no embedder was prepared")
            steps.create_indexes(enrichment.dimensions, enrichment.name)
            report = steps.embed(db_id, enrichment.embedder, enrichment.name, reembed=options.reembed)
            result.embedded_tables, result.embedded_columns = report.tables, report.columns
            result.tables_current, result.tables_total = report.tables_current, report.tables_total
            result.columns_current, result.columns_total = report.columns_current, report.columns_total
            result.partial = report.stopped
            result.steps_done.append("embed" if report.stopped is None else "embed (partial)")

        result.problems = list(steps.verify(meta))
        result.steps_done.append("verify")
    except Exception as exc:  # reported per database, not fatal to the batch
        log.debug("graph build failed for %s", db_id, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.monotonic() - start
    return result


def build_many(
    db_ids: list[str],
    options: Options | None = None,
    enrichment: Enrichment | None = None,
    steps: Steps | None = None,
) -> list[GraphResult]:
    options = options or Options()
    steps = steps or default_steps()
    return [build_graph(d, options, enrichment, steps) for d in db_ids]


def format_results(results: list[GraphResult], options: Options | None = None) -> str:
    options = options or Options()
    lines: list[str] = []
    for r in results:
        if r.error:
            lines.append(f"  FAIL  {r.db_id:<28} after [{', '.join(r.steps_done) or 'nothing'}]: {r.error}")
            continue
        edges = f"{r.declared} declared + {r.inferred} inferred relationships"
        embed = ""
        if not options.skip_enrichment:
            embed = (
                f", embedded {r.tables_current}/{r.tables_total} tables and {r.columns_current}/{r.columns_total} columns"
                f" ({r.embedded_tables + r.embedded_columns} new)"
            )
        status = "DIFF" if r.problems else "PART" if r.partial else "ok  "
        lines.append(f"  {status}  {r.db_id:<28} {r.tables:>4} tables {r.columns:>5} columns, {edges}{embed}  ({r.seconds:.1f}s)")
        if r.partial:
            lines.append(f"          stopped early: {r.partial}")
            lines.append(
                "          what was embedded is saved; rerun the same command"
                + (" WITHOUT --rebuild (it would clear it)" if options.rebuild else "")
                + " later to finish (add --skip-extraction to reuse the metadata file)"
            )
        for problem in r.problems[:10]:
            lines.append(f"          {problem}")
        if len(r.problems) > 10:
            lines.append(f"          ... and {len(r.problems) - 10} more")
        if r.problems and not options.rebuild:
            lines.append("          the graph differs from the metadata; if tables were dropped or renamed, rerun with --rebuild")
    failed = sum(r.failed for r in results)
    partial = sum(r.partial is not None and not r.failed for r in results)
    built = len(results) - failed - partial
    lines.append(f"{built} built, {partial} partial, {failed} failed")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, steps: Steps | None = None) -> int:
    from schemagraph.extraction import infer

    parser = argparse.ArgumentParser(description="Extract, infer, ingest, weight, embed and verify a database's graph.")
    parser.add_argument("--db", action="append", default=[], help="db_id (repeatable)")
    parser.add_argument("--all", action="store_true", help="every database the read-only role can see")
    parser.add_argument("--rebuild", action="store_true", help="delete each db_id's nodes before ingesting (drops stale tables and embeddings)")
    parser.add_argument("--skip-extraction", action="store_true", help="reuse data/metadata/<db>.json")
    parser.add_argument("--skip-inference", action="store_true", help="declared relationships only")
    parser.add_argument("--skip-enrichment", action="store_true", help="no vector indexes or embeddings")
    parser.add_argument("--reembed", action="store_true", help="embed every node again, not only new or changed ones")
    parser.add_argument("--sample-n", type=int, default=5)
    parser.add_argument("--min-overlap", type=float, default=infer.MIN_OVERLAP)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if bool(args.db) == args.all:
        parser.error("give one or more --db, or --all (not both)")
    if args.skip_extraction and args.skip_inference:
        parser.error("--skip-inference has no effect with --skip-extraction (the metadata file is reused as it is)")

    options = Options(
        rebuild=args.rebuild,
        skip_extraction=args.skip_extraction,
        skip_inference=args.skip_inference,
        skip_enrichment=args.skip_enrichment,
        reembed=args.reembed,
        sample_n=args.sample_n,
        min_overlap=args.min_overlap,
        min_confidence=args.min_confidence,
    )
    steps = steps or default_steps()

    from schemagraph.graph import driver

    try:
        if args.all:
            from schemagraph import db

            db_ids = db.list_databases()
        else:
            db_ids = list(dict.fromkeys(args.db))
        if not db_ids:
            parser.error("no databases found")
        driver.check_connection()
        enrichment = None if options.skip_enrichment else prepare_enrichment()
        results = build_many(db_ids, options, enrichment, steps)
    except SetupError as exc:
        print(f"cannot start: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - Neo4j or Postgres unreachable
        print(f"cannot start: {type(exc).__name__}: {exc}")
        return 2
    finally:
        driver.close_driver()

    print(format_results(results, options))
    if any(r.failed for r in results):
        return 1
    return 3 if any(r.partial for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
