"""Shared Pydantic data contracts for SchemaGraph (SPEC.md section 3).

Changes to this file go through a pull request reviewed by the whole group.
Every change here must also update the matching sample in contracts/examples/.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared literals
# ---------------------------------------------------------------------------

Mode = Literal["full", "retrieval", "graph"]
DescriptionSource = Literal["catalog", "generated"]  # None = no description
RelationshipSource = Literal["declared", "inferred"]
Cardinality = Literal["N:1", "1:1"]
Traversal = Literal["many_to_one", "one_to_many"]
PlanStatus = Literal["full", "partial", "single_table", "none"]
PlanAlgorithm = Literal["gds_steiner", "exact_steiner", "networkx_steiner", "shortest_path", "none"]
AskStatus = Literal[
    "success", "invalid_sql", "execution_error", "no_schema", "unsafe_sql"
]
TableRole = Literal["seed", "bridge", "context"]


class Contract(BaseModel):
    """Base class: strict about unknown fields so contract drift is caught early."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# UID helpers (SPEC 2.1). Everyone builds identifiers through these functions.
# Names are used exactly as stored in the PostgreSQL catalog.
# ---------------------------------------------------------------------------


def table_uid(db_id: str, schema: str, table: str) -> str:
    return f"{db_id}:{schema}.{table}"


def column_uid(db_id: str, schema: str, table: str, column: str) -> str:
    return f"{db_id}:{schema}.{table}.{column}"


def schema_uid(db_id: str, schema: str) -> str:
    return f"{db_id}:{schema}"


def table_uid_of_column(col_uid: str) -> str:
    """'db:schema.table.column' -> 'db:schema.table'."""
    return col_uid.rsplit(".", 1)[0]


def display_name(uid: str) -> str:
    """Strip the db_id prefix: 'db:sales.customer.id' -> 'sales.customer.id'."""
    return uid.split(":", 1)[1] if ":" in uid else uid


def split_table_uid(uid: str) -> tuple[str, str, str]:
    """'db:schema.table' -> (db_id, schema, table)."""
    db_id, rest = uid.split(":", 1)
    schema, table = rest.split(".", 1)
    return db_id, schema, table


# ---------------------------------------------------------------------------
# 3.1 Database metadata
# ---------------------------------------------------------------------------


class SchemaMeta(Contract):
    name: str


class TableMeta(Contract):
    uid: str
    schema_: str = Field(alias="schema")  # "schema" shadows a BaseModel attribute
    name: str
    description: str | None = None
    description_source: DescriptionSource | None = None
    row_count: int | None = None
    primary_key: list[str] = Field(default_factory=list)  # in key order
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ColumnMeta(Contract):
    uid: str
    table_uid: str
    name: str
    data_type: str
    ordinal_position: int
    is_nullable: bool
    is_primary_key: bool = False
    is_unique: bool = False  # unique on its own (not just inside a composite key)
    description: str | None = None
    description_source: DescriptionSource | None = None
    sample_values: list[str] = Field(default_factory=list)  # up to 5, as strings


class InferenceEvidence(Contract):
    name_match: Literal["exact", "suffix", "table_name"] | None = None
    type_compatible: bool
    value_overlap: float = Field(ge=0.0, le=1.0)


class RelationshipMeta(Contract):
    """Column-to-column link, always FK column -> referenced column."""

    from_column: str
    to_column: str
    source: RelationshipSource
    confidence: float = Field(ge=0.0, le=1.0)  # 1.0 for declared
    cardinality: Cardinality
    constraint_name: str | None = None  # None for inferred
    evidence: InferenceEvidence | None = None  # None for declared


class DatabaseMetadata(Contract):
    db_id: str
    extracted_at: datetime
    schemas: list[SchemaMeta]
    tables: list[TableMeta]
    columns: list[ColumnMeta]
    relationships: list[RelationshipMeta]

    def check_integrity(self) -> list[str]:
        """Return human-readable problems (empty list = OK)."""
        problems: list[str] = []
        table_uids = {t.uid for t in self.tables}
        column_uids = {c.uid for c in self.columns}
        schema_names = {s.name for s in self.schemas}

        for t in self.tables:
            if t.schema_ not in schema_names:
                problems.append(f"table {t.uid}: schema '{t.schema_}' not in schemas")
        for c in self.columns:
            if c.table_uid not in table_uids:
                problems.append(f"column {c.uid}: unknown table {c.table_uid}")
        for r in self.relationships:
            for end in (r.from_column, r.to_column):
                if end not in column_uids:
                    problems.append(f"relationship endpoint missing: {end}")
            if r.source == "declared" and r.confidence != 1.0:
                problems.append(f"declared relationship {r.from_column} has confidence != 1.0")
        return problems


# ---------------------------------------------------------------------------
# 3.2 Seed tables (retrieval output)
# ---------------------------------------------------------------------------


class SeedTable(Contract):
    uid: str
    score: float
    matched_on: str  # "table" or "column:<name>"


# ---------------------------------------------------------------------------
# 3.3 Join plan (planner output)
# ---------------------------------------------------------------------------


class PlannedJoin(Contract):
    # Always in stored direction (FK -> referenced), regardless of traversal.
    from_column: str
    to_column: str
    source: RelationshipSource
    confidence: float
    cardinality: Cardinality
    traversal: Traversal
    weight: float


class PlanAlternative(Contract):
    tables: list[str]
    total_cost: float


class JoinPlan(Contract):
    db_id: str
    status: PlanStatus
    algorithm: PlanAlgorithm
    seed_tables: list[SeedTable] = Field(default_factory=list)
    bridge_tables: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    joins: list[PlannedJoin] = Field(default_factory=list)
    total_cost: float = 0.0
    unreachable_seeds: list[str] = Field(default_factory=list)
    dropped_seeds: list[str] = Field(default_factory=list)
    alternatives: list[PlanAlternative] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3.4 Schema context (what goes into the prompt)
# ---------------------------------------------------------------------------


class SchemaContext(Contract):
    db_id: str
    mode: Mode
    tables: list[str]
    ddl: str
    join_text: str = ""  # empty in full / retrieval modes
    join_plan: JoinPlan | None = None  # None in full / retrieval modes
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3.5 API request and response
# ---------------------------------------------------------------------------


class AskOptions(Contract):
    top_k: int | None = None
    max_attempts: int | None = None
    row_limit: int | None = None


class AskRequest(Contract):
    question: str = Field(min_length=1)
    db_id: str
    mode: Mode = "graph"
    options: AskOptions | None = None  # defaults come from config.py


class QueryResult(Contract):
    columns: list[str]
    rows: list[list[str | None]]  # every value a string or null
    row_count: int
    truncated: bool = False


class ExplainTable(Contract):
    uid: str
    display_name: str
    role: TableRole
    reason: str


class ExplainJoin(Contract):
    from_: str = Field(alias="from")
    to: str
    source: RelationshipSource
    text: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Explanation(Contract):
    summary: str
    tables: list[ExplainTable] = Field(default_factory=list)
    join_path: list[ExplainJoin] = Field(default_factory=list)
    interpretation: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Attempt(Contract):
    sql: str
    error: str | None = None


class LatencyMs(Contract):
    retrieval: int = 0
    planning: int = 0
    generation: int = 0
    validation: int = 0
    execution: int = 0
    explanation: int = 0
    total: int = 0


class Metrics(Contract):
    input_tokens: int = 0
    output_tokens: int = 0
    attempt_count: int = 0
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)


class AskResponse(Contract):
    request_id: str
    question: str
    db_id: str
    mode: Mode
    status: AskStatus
    sql: str | None = None
    result: QueryResult | None = None
    explanation: Explanation | None = None
    join_plan: JoinPlan | None = None  # None outside graph mode
    attempts: list[Attempt] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)


# ---------------------------------------------------------------------------
# 3.6 / 3.7 Evaluation files
# ---------------------------------------------------------------------------


class Question(Contract):
    id: str
    db_id: str
    question: str
    gold_sql: str
    tier: Literal[1, 2, 3]
    split: Literal["test", "dev"]
    order_matters: bool = False
    evidence: str | None = None  # BIRD hint; None for AdventureWorks
    author: str = ""
    notes: str = ""


class FewShotExample(Question):
    """Same format as Question. Must never contain test questions."""


class EvalResult(Contract):
    question_id: str
    db_id: str
    mode: Mode
    tier: Literal[1, 2, 3]
    correct: bool
    status: AskStatus
    generated_sql: str | None
    gold_tables: list[str]
    context_tables: list[str]
    schema_recall: float
    bridge_tables_needed: list[str]
    bridge_tables_recovered: list[str]
    plan_status: PlanStatus | None = None
    input_tokens: int = 0
    latency_ms: int = 0
    attempt_count: int = 0
    llm_model: str
    run_id: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Supporting types named in SPEC section 5 signatures but not defined in
# section 3. These are DRAFTS; flag them in the PR for the owners.
# ---------------------------------------------------------------------------


class ExplainResult(Contract):  # db/postgres.py explain()
    ok: bool
    error: str | None = None


class CandidatePair(Contract):  # extraction/infer.py
    from_column: str
    to_column: str
    name_match: Literal["exact", "suffix", "table_name"] | None = None
    type_compatible: bool


class IngestReport(Contract):  # graph/ingest.py
    db_id: str
    schemas: int = 0
    tables: int = 0
    columns: int = 0
    contains_edges: int = 0
    has_column_edges: int = 0
    references_edges: int = 0
    warnings: list[str] = Field(default_factory=list)


class GenerationResult(Contract):  # pipeline/generate.py
    sql: str | None
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0


class TableDetail(Contract):  # graph/queries.py: table + columns, for DDL
    table: TableMeta
    columns: list[ColumnMeta]
    references: list[RelationshipMeta] = Field(default_factory=list)  # outgoing FKs


class TableSummary(Contract):  # /api/schema
    uid: str
    display_name: str
    description: str | None = None
    column_count: int


class TableWithRelationships(Contract):  # /api/schema/{db_id}/tables/{uid}
    table: TableMeta
    columns: list[ColumnMeta]
    references: list[RelationshipMeta] = Field(default_factory=list)
    referenced_by: list[RelationshipMeta] = Field(default_factory=list)
