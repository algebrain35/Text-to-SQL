# SchemaGraph — Codebase Specification

This document defines how the codebase is organized, what each module exposes, the API endpoints, and the JSON formats passed between components. Its purpose is to let all four members build in parallel against agreed interfaces.

**Status:** this version reflects the implemented code. Where the first draft and the code differed, the spec now follows the code; each such change is marked **(changed)** and collected in Section 8, which also has the build status of every module. Items marked **(open)** await a team decision.

**Rule:** the data contracts in Section 3 (implemented in `schemagraph/models.py`) are shared. Any change to them goes through a pull request that the whole group reviews, because every component depends on them.

---

## 1. Repository layout

```
schemagraph/
├── docker-compose.yml          # PostgreSQL, Neo4j Community (no GDS plugin), API
├── config.example.yml          # copy to config.yml: Postgres roles, Neo4j, embedding   (changed from .env.example)
├── pyproject.toml              # dependencies and pytest config                           (changed from requirements.txt)
├── pytest.ini                  # markers: integration, adventureworks, neo4j
├── README.md                   # setup, data sources, API reference, who owns what
├── SPEC.md                     # this file
│
├── contracts/
│   └── examples/               # sample JSON for every contract (used as mocks)
│       ├── metadata.adventureworks.sample.json
│       ├── join_plan.sample.json
│       └── ask_response.sample.json
│
├── schemagraph/                # Python package (backend)
│   ├── config.py               # settings loaded from config.yml    [shared]
│   ├── models.py               # Pydantic data contracts            [shared]
│   │
│   ├── db/                     # PostgreSQL access                   [Emmanuel]
│   │   └── postgres.py
│   │
│   ├── extraction/             # catalog extraction + inference      [Emmanuel]
│   │   ├── sql/
│   │   │   ├── tables.sql
│   │   │   ├── columns.sql
│   │   │   ├── primary_keys.sql
│   │   │   ├── foreign_keys.sql
│   │   │   └── unique_constraints.sql
│   │   ├── extract.py
│   │   ├── infer.py
│   │   └── build_metadata.py   # CLI: extract → infer → save, for one or all databases
│   │
│   ├── graph/                  # Neo4j model, enrichment, planning   [Cameron]
│   │   ├── driver.py           # one shared Neo4j driver and session()
│   │   ├── cypher.py           # session helpers shared by ingest/queries/retrieve
│   │   ├── ingest.py
│   │   ├── weights.py
│   │   ├── vector.py           # Embedder interface, text to embed, vector indexes
│   │   ├── enrich.py
│   │   ├── queries.py
│   │   ├── retrieve.py
│   │   └── planner.py
│   │
│   ├── pipeline/               # LangChain pipeline                  [Mohammed]
│   │   ├── context.py
│   │   ├── examples.py
│   │   ├── generate.py
│   │   ├── validate.py
│   │   ├── explain.py
│   │   └── pipeline.py
│   │
│   ├── api/                    # FastAPI app                          [Mohammed]
│   │   └── main.py
│   │
│   └── eval/                   # evaluation harness                   [Emmanuel]
│       ├── questions/
│       │   ├── adventureworks.jsonl
│       │   ├── bird_subset.jsonl
│       │   └── fewshot_examples.jsonl
│       ├── infer_recovery.py   # hides declared FKs, measures how many inference recovers
│       ├── compare.py
│       ├── schema_metrics.py
│       └── run_eval.py
│
├── scripts/
│   ├── create_roles.sql        # sg_readonly role and grants (Section 7)
│   ├── load_adventureworks.sh  # downloads/loads AdventureWorks into Postgres
│   ├── load_bird.sh            # loads the BIRD mini-dev PostgreSQL databases
│   ├── build_graph.py          # CLI: extract → infer → ingest → enrich
│   └── export_contracts.py     # writes JSON Schema from models.py
│
├── tests/                      # pytest, one file per module
│
└── frontend/                   # React + Vite                          [Davin]
    └── src/
        ├── api/
        │   ├── client.ts
        │   └── types.ts        # generated from the API's OpenAPI schema
        ├── components/
        └── mocks/              # copies of contracts/examples/*.json
```

Data files (`data/metadata/{db_id}.json`, `data/bird/`) are generated and not committed.

**Ports:** PostgreSQL 5432 · Neo4j 7474 (browser) / 7687 (bolt) · API 8000 · frontend 5173.

---

## 2. Conventions

### 2.1 Identifiers

Several databases (AdventureWorks plus a few BIRD databases) share one Neo4j instance. Neo4j Community Edition supports only one user database, so every node carries a `db_id` and every identifier is prefixed with it.

| Thing | Format | Example |
|---|---|---|
| Database ID (`db_id`) | lowercase letters, digits and `_` (`^[a-z0-9_]+$`); equals the PostgreSQL database name | `adventureworks` |
| Schema UID | `{db_id}:{schema}` | `adventureworks:sales` **(open: confirm with Cameron)** |
| Table UID | `{db_id}:{schema}.{table}` | `adventureworks:sales.customer` |
| Column UID | `{db_id}:{schema}.{table}.{column}` | `adventureworks:sales.customer.customerid` |
| Display name (shown to users) | `{schema}.{table}[.{column}]` | `sales.customer.customerid` |

- Use names exactly as stored in the PostgreSQL catalog (PostgreSQL folds unquoted names to lowercase).
- Build UIDs only with the helpers in `models.py`, and never parse one: use `ColumnMeta.table_uid` to get a column's table.
- Dump models with `by_alias=True` so the aliases (`schema` on tables, `from` on `ExplainJoin`) are written.
- Uniqueness constraints in Neo4j are on `uid`, never on bare names (bare names repeat across tables and schemas).

### 2.2 Relationship direction and cardinality

- A `REFERENCES` relationship always points **from the foreign-key column to the referenced column** (the "many" side to the "one" side).
- `cardinality` is `"N:1"`, or `"1:1"` when the foreign-key column is itself unique.
- A `1:1` edge is safe to walk both ways, so it costs the same in both directions **(changed; open)**.
- Traversing a relationship in its stored direction is **many-to-one** (safe). Traversing against it is **one-to-many** (fan-out: can duplicate rows).

### 2.3 Modes

Every pipeline call takes a `mode`, which selects how schema context is built. All three share the same LLM, prompt template, few-shot examples, and validation loop.

| Mode | Schema context |
|---|---|
| `full` | DDL for every table in the database |
| `retrieval` | DDL for vector-search seed tables only, no joins |
| `graph` | Seed tables + bridge tables + explicit join conditions from the join plan |

### 2.4 General

- Python 3.11+ (developed on 3.13), type hints everywhere, Pydantic v2 for all data passed between modules.
- Functions return Pydantic models, not raw dicts.
- LLM temperature is always 0. Model names come from `config.py`, never hardcoded.
- Every module has a pytest file. Pure functions (weights, result comparison, SQL table extraction) must have unit tests.

---

## 3. Data contracts (JSON formats)

All of these are defined as Pydantic models in `schemagraph/models.py`. `scripts/export_contracts.py` writes their JSON Schemas so the frontend and tests can validate against them.

### 3.1 Database metadata — extraction output, graph ingestion input

Produced by `extraction/extract.py` + `extraction/infer.py`. Consumed by `graph/ingest.py`. Saved as `data/metadata/{db_id}.json`.

```json
{
  "db_id": "adventureworks",
  "extracted_at": "2026-10-22T14:03:00Z",
  "schemas": [
    { "name": "sales" },
    { "name": "person" }
  ],
  "tables": [
    {
      "uid": "adventureworks:sales.customer",
      "schema": "sales",
      "name": "customer",
      "description": "Current customer information.",
      "description_source": "catalog",
      "row_count": 19820,
      "primary_key": ["customerid"]
    }
  ],
  "columns": [
    {
      "uid": "adventureworks:sales.customer.customerid",
      "table_uid": "adventureworks:sales.customer",
      "name": "customerid",
      "data_type": "integer",
      "ordinal_position": 1,
      "is_nullable": false,
      "is_primary_key": true,
      "is_unique": true,
      "description": "Primary key.",
      "description_source": "catalog",
      "sample_values": ["1", "2", "3", "4", "5"]
    }
  ],
  "relationships": [
    {
      "from_column": "adventureworks:sales.salesorderheader.customerid",
      "to_column": "adventureworks:sales.customer.customerid",
      "source": "declared",
      "confidence": 1.0,
      "cardinality": "N:1",
      "constraint_name": "fk_salesorderheader_customer_customerid",
      "evidence": null
    },
    {
      "from_column": "adventureworks:sales.store.salespersonid",
      "to_column": "adventureworks:sales.salesperson.businessentityid",
      "source": "inferred",
      "confidence": 0.72,
      "cardinality": "N:1",
      "constraint_name": null,
      "evidence": { "name_match": "suffix", "type_compatible": true, "value_overlap": 0.97 }
    }
  ]
}
```

| Field | Values / notes |
|---|---|
| `description_source` | `"catalog"` (from `obj_description`/`col_description`), `"generated"` (set later by enrichment), or `null` if none |
| `sample_values` | Up to 5 distinct non-null values as strings; used for inference, for generating descriptions, and (first 3, trimmed to 40 characters) in the column's embedding text |
| `primary_key` | Column **names** in key order, not UIDs |
| `confidence` (inferred) | `0.4 * name_score + 0.6 * value_overlap`, capped at 0.95, so an inferred edge always costs more than a declared one |
| `source` | `"declared"` (from constraint views) or `"inferred"` |
| `confidence` | `1.0` for declared; 0–1 for inferred |
| `evidence` | `null` for declared; inference signals for inferred |

### 3.2 Seed tables — retrieval output

```json
[
  { "uid": "adventureworks:sales.salesorderheader", "score": 0.84, "matched_on": "column:totaldue" },
  { "uid": "adventureworks:person.person", "score": 0.79, "matched_on": "column:firstname" }
]
```

`matched_on` records whether the table matched on its own description (`"table"`) or through one of its columns (`"column:<name>"`). It is used in explanations.

### 3.3 Join plan — planner output

Produced by `graph/planner.py` (graph mode only). Consumed by `pipeline/context.py`, `pipeline/explain.py`, and evaluation.

```json
{
  "db_id": "adventureworks",
  "status": "full",
  "algorithm": "shortest_path",
  "seed_tables": [
    { "uid": "adventureworks:sales.salesorderheader", "score": 0.84, "matched_on": "column:totaldue" },
    { "uid": "adventureworks:person.person", "score": 0.79, "matched_on": "column:firstname" }
  ],
  "bridge_tables": ["adventureworks:sales.customer"],
  "tables": [
    "adventureworks:sales.salesorderheader",
    "adventureworks:sales.customer",
    "adventureworks:person.person"
  ],
  "joins": [
    {
      "from_column": "adventureworks:sales.salesorderheader.customerid",
      "to_column": "adventureworks:sales.customer.customerid",
      "source": "declared",
      "confidence": 1.0,
      "cardinality": "N:1",
      "traversal": "many_to_one",
      "weight": 1.0
    },
    {
      "from_column": "adventureworks:sales.customer.personid",
      "to_column": "adventureworks:person.person.businessentityid",
      "source": "declared",
      "confidence": 1.0,
      "cardinality": "N:1",
      "traversal": "many_to_one",
      "weight": 1.0
    }
  ],
  "total_cost": 2.0,
  "unreachable_seeds": [],
  "dropped_seeds": [],
  "alternatives": [
    {
      "tables": [
        "adventureworks:sales.salesorderheader",
        "adventureworks:sales.salesperson",
        "adventureworks:humanresources.employee",
        "adventureworks:person.person"
      ],
      "total_cost": 3.0
    }
  ]
}
```

| Field | Values / notes |
|---|---|
| `status` | `"full"` (all seeds connected), `"partial"` (some seeds unreachable), `"single_table"` (one seed, no joins), `"none"` (no plan; pipeline falls back to retrieval context) |
| `algorithm` | `"exact_steiner"` (directed Dreyfus–Wagner, up to 7 seeds), `"shortest_path"` (two seeds), `"networkx_steiner"` (fallback for more seeds, or on request), `"none"`. `"gds_steiner"` stays in the contract but is reserved and never produced **(changed)** |
| `unreachable_seeds` | Seeds with no path to the others. Listed here, and excluded from `tables` **(open)** |
| `joins[].from_column` / `to_column` | Always in stored relationship direction (FK → referenced column), regardless of traversal |
| `traversal` | `"many_to_one"` or `"one_to_many"` (fan-out) |
| `dropped_seeds` | Seeds removed because the tree exceeded the cost/bridge threshold |
| `alternatives` | Optional runner-up paths, used to tell the user how an ambiguous question was interpreted |

### 3.4 Schema context — what goes into the prompt

Internal to the pipeline, built by `pipeline/context.py` for every mode.

```json
{
  "db_id": "adventureworks",
  "mode": "graph",
  "tables": ["adventureworks:sales.salesorderheader", "adventureworks:sales.customer", "adventureworks:person.person"],
  "ddl": "CREATE TABLE sales.salesorderheader (...);\nCREATE TABLE sales.customer (...);\n...",
  "join_text": "sales.salesorderheader.customerid = sales.customer.customerid\nsales.customer.personid = person.person.businessentityid",
  "join_plan": { "...": "JoinPlan or null" },
  "notes": []
}
```

`join_text` is empty and `join_plan` is `null` in `full` and `retrieval` modes. `notes` carries warnings to include in the prompt (e.g., a table with no known relationship).

### 3.5 API request and response — `POST /api/ask`

**Request**

```json
{
  "question": "What is the total order value for each customer's name?",
  "db_id": "adventureworks",
  "mode": "graph",
  "options": { "top_k": 5, "max_attempts": 3, "row_limit": 200 }
}
```

`mode` defaults to `"graph"`. `options` is optional; defaults come from `config.py`.

**Response**

```json
{
  "request_id": "6f1c2a0e-9b7d-4c1e-8a34-2d5e9f0b7c11",
  "question": "What is the total order value for each customer's name?",
  "db_id": "adventureworks",
  "mode": "graph",
  "status": "success",
  "sql": "SELECT p.firstname, p.lastname, SUM(soh.totaldue) AS total_order_value FROM sales.salesorderheader soh JOIN sales.customer c ON soh.customerid = c.customerid JOIN person.person p ON c.personid = p.businessentityid GROUP BY p.firstname, p.lastname;",
  "result": {
    "columns": ["firstname", "lastname", "total_order_value"],
    "rows": [["Jon", "Yang", "8248.99"], ["Eugene", "Huang", "6383.66"]],
    "row_count": 2,
    "truncated": false
  },
  "explanation": {
    "summary": "This adds up the total value of all orders for each customer and shows it next to the customer's name.",
    "tables": [
      { "uid": "adventureworks:sales.salesorderheader", "display_name": "sales.salesorderheader", "role": "seed", "reason": "Contains order totals (totaldue)." },
      { "uid": "adventureworks:sales.customer", "display_name": "sales.customer", "role": "bridge", "reason": "Needed to connect orders to people." },
      { "uid": "adventureworks:person.person", "display_name": "person.person", "role": "seed", "reason": "Contains first and last names." }
    ],
    "join_path": [
      { "from": "sales.salesorderheader.customerid", "to": "sales.customer.customerid", "source": "declared", "text": "Each order belongs to one customer." },
      { "from": "sales.customer.personid", "to": "person.person.businessentityid", "source": "declared", "text": "Each customer is linked to one person record." }
    ],
    "interpretation": "\"Person\" was interpreted as the customer, not the salesperson.",
    "warnings": []
  },
  "join_plan": { "...": "full JoinPlan from 3.3, or null outside graph mode" },
  "attempts": [
    { "sql": "SELECT ...", "error": null }
  ],
  "metrics": {
    "input_tokens": 1834,
    "output_tokens": 212,
    "attempt_count": 1,
    "latency_ms": { "retrieval": 120, "planning": 45, "generation": 2310, "validation": 18, "execution": 32, "explanation": 1400, "total": 3925 }
  }
}
```

| Field | Values / notes |
|---|---|
| `status` | `"success"`, `"invalid_sql"` (failed validation after all attempts), `"execution_error"`, `"no_schema"` (nothing relevant retrieved), `"unsafe_sql"` (not a single SELECT) |
| `sql` | Final SQL, or `null` if none was produced |
| `result.rows` | All values serialized as strings or `null` (avoids date/decimal JSON issues). Capped at `row_limit`; `truncated` is true if more rows existed |
| `explanation.tables[].role` | `"seed"`, `"bridge"`, or `"context"` (full/retrieval modes) |
| `attempts` | Every generated SQL, in order, with the database error if it failed |

**HTTP status codes:** `200` for every completed pipeline run, including failures reported in `status`, so the frontend can still show the explanation and attempts. `400` for bad input (empty question, unknown `db_id`, invalid mode). `503` if PostgreSQL, Neo4j, or the LLM is unreachable. Errors use FastAPI's default `{"detail": "..."}` body.

### 3.6 Evaluation question file — JSONL, one question per line

```json
{"id": "aw-017", "db_id": "adventureworks", "question": "What is the total order value for each customer's name?", "gold_sql": "SELECT ...", "tier": 3, "split": "test", "order_matters": false, "evidence": null, "author": "cameron", "notes": ""}
```

| Field | Values / notes |
|---|---|
| `tier` | `1` single table · `2` joins within one schema · `3` cross-schema joins of 3+ tables |
| `split` | `"test"` (the ~50 scored questions) or `"dev"` (~10 used only for tuning weights and prompts) |
| `order_matters` | `true` only if the question asks for an ordering (e.g., "top 10") |
| `evidence` | BIRD's external-knowledge hint, if any; `null` for AdventureWorks |

`fewshot_examples.jsonl` uses the same format but holds the example pairs shown to the LLM. **It must never contain test questions.**

### 3.7 Evaluation result file — JSONL, one row per question per mode

```json
{"question_id": "aw-017", "db_id": "adventureworks", "mode": "graph", "tier": 3, "correct": true, "status": "success", "generated_sql": "SELECT ...", "gold_tables": ["sales.salesorderheader", "sales.customer", "person.person"], "context_tables": ["sales.salesorderheader", "sales.customer", "person.person"], "schema_recall": 1.0, "bridge_tables_needed": ["sales.customer"], "bridge_tables_recovered": ["sales.customer"], "plan_status": "full", "input_tokens": 1834, "latency_ms": 3925, "attempt_count": 1, "llm_model": "<model>", "run_id": "2026-11-12-a", "timestamp": "2026-11-12T19:22:10Z"}
```

`bridge_tables_needed` = tables in the gold SQL that were not among the seed tables.

---

## 4. API endpoints

Defined in `schemagraph/api/main.py`. FastAPI generates OpenAPI docs at `/docs`; the frontend generates `types.ts` from `/openapi.json`.

| Method | Path | Purpose | Response |
|---|---|---|---|
| `GET` | `/api/health` | Check PostgreSQL, Neo4j, and LLM connectivity | `{"postgres": "ok", "neo4j": "ok", "llm": "ok"}` |
| `GET` | `/api/databases` | List databases available to query (for the frontend selector) | `[{"db_id": "adventureworks", "table_count": 68, "schema_count": 5}]` |
| `POST` | `/api/ask` | Main endpoint: question → SQL, result, explanation | `AskResponse` (3.5) |
| `POST` | `/api/plan` | Retrieval + join planning only, no LLM call. For debugging and for testing the graph independently | `{"seed_tables": [...], "join_plan": {...}}` |
| `GET` | `/api/schema/{db_id}/tables` | Browse tables (onboarding feature) | `[{"uid", "display_name", "description", "column_count"}]` |
| `GET` | `/api/schema/{db_id}/tables/{table_uid}` | One table with its columns and relationships | `{"table": {...}, "columns": [...], "references": [...], "referenced_by": [...]}` |

`/api/plan` takes the same request body as `/api/ask`. The two `/api/schema` endpoints are lower priority; build them after `/api/ask` works.

Graph building is **not** an endpoint. It runs from the command line with `scripts/build_graph.py`.

---

## 5. Modules and methods

Signatures are the contract; internal helper functions are up to each owner.

### 5.1 `config.py` — shared  **(changed: YAML instead of `.env`)**

Settings come from `config.yml` (path overridable with `SCHEMAGRAPH_CONFIG`), validated by Pydantic. Secrets are never in the file: each section names an environment variable that holds the password or key.

```yaml
postgres:
  host: localhost
  port: 5432
  admin:    { user: sg_admin,    password_env: PG_ADMIN_PASSWORD }      # extraction, inference, loading
  readonly: { user: sg_readonly, password_env: PG_READONLY_PASSWORD }   # all LLM SQL
  statement_timeout_ms: 10000
neo4j:                           # optional; needed by graph/
  uri: bolt://localhost:7687
  user: neo4j
  password_env: NEO4J_PASSWORD
  database: neo4j                # Community has one database; every db_id lives in it
embedding:                       # optional; needed by enrich/retrieve
  base_url: https://api.openai.com/v1
  model: <embedding model name>  # team's choice
  dimensions: <vector size>      # must match the vector index; changing it means re-embedding
  api_key_env: OPENAI_API_KEY    # null for a local server with no key
  send_dimensions: false
  batch_size: 64
  timeout_s: 30
# llm:      <- with pipeline/ (model name, temperature fixed at 0)
# pipeline: <- top_k (5), max_attempts (3), row_limit (200)
```

```python
def get_config() -> AppConfig          # cached; load_config(path) loads a given file
class PostgresConfig: conninfo(role, db_id)   # role is "admin" or "readonly"; client_encoding=utf8
class Neo4jConfig; class EmbeddingConfig   # .password / .api_key read the env var, raising if unset
```

The planner's tuning values are module constants rather than settings **(changed)**: `CONFIDENCE_PENALTY = 5.0` and `FAN_OUT_PENALTY = 1.0` in `graph/weights.py`, `MAX_BRIDGE_TABLES = 4` and `MAX_EXACT_SEEDS = 7` in `graph/planner.py`. Move them into config if experiments need to vary them.

### 5.2 `db/postgres.py` — Emmanuel

```python
def list_databases() -> list[str]
def run_readonly_query(db_id: str, sql: str, row_limit: int) -> QueryResult
    # read-only role, statement_timeout set, rows serialized as strings
def explain(db_id: str, sql: str) -> ExplainResult        # ExplainResult(ok: bool, error: str | None)
def run_admin_query(db_id: str, sql: str, params: dict | None = None) -> list[dict]
    # extraction and inference only, never for LLM-generated SQL
```

Read-only connections use the `sg_readonly` role with `statement_timeout` and read-only transactions.

### 5.3 `extraction/extract.py` — Emmanuel

```python
def extract_metadata(db_id: str, *, sample_n: int = 5) -> DatabaseMetadata
def fetch_tables(db_id: str) -> list[TableMeta]
def fetch_columns(db_id: str) -> list[ColumnMeta]
def fetch_primary_keys(db_id: str) -> dict[str, list[str]]      # table_uid -> column names
def fetch_foreign_keys(db_id: str) -> list[RelationshipMeta]     # source="declared"
def fetch_unique_columns(db_id: str) -> set[str]                 # column UIDs
def fetch_sample_values(db_id: str, column_uid: str, n: int = 5) -> list[str]
def save_metadata(meta: DatabaseMetadata, path: Path) -> None
def load_metadata(path: Path) -> DatabaseMetadata
```

SQL lives in `extraction/sql/*.sql` so it can be read and graded on its own. CLI: `python -m schemagraph.extraction.extract --db X [--out PATH]`.

`extraction/build_metadata.py` runs extract → infer → save for one database (`--db`, repeatable) or all (`--all`), with `--out`, `--skip-inference`, `--sample-n`, `--min-overlap` and `--min-confidence`. It exits non-zero if any database failed.

### 5.4 `extraction/infer.py` — Emmanuel

```python
def find_candidate_pairs(meta: DatabaseMetadata) -> list[CandidatePair]
    # name matching + compatible types, skipping pairs already declared
def value_overlap(db_id: str, from_column: str, to_column: str) -> float
    # share of distinct from_column values that exist in to_column
def infer_relationships(db_id: str, meta: DatabaseMetadata, min_confidence: float = 0.5,
                        min_overlap: float = 0.9) -> list[RelationshipMeta]
    # source="inferred", with confidence and evidence
def with_inferred(db_id: str, meta: DatabaseMetadata, **kwargs) -> DatabaseMetadata
    # replaces any old inferred edges; idempotent
```

Rules, in order: (1) the target column is unique; (2) the name signal is `exact`, `suffix` or `table_name` (generic names such as `id` and `rowguid` never match); (3) types are in the same family; (4) value overlap is at least `min_overlap`; (5) each source column gets one parent, the tightest match; (6) a unique source (1:1) needs an `exact` name and must point from the smaller table to the larger, and an edge is never inferred in both directions; (7) declared relationships are never replaced. `confidence = 0.4 * name_score + 0.6 * overlap`, capped at 0.95. Tuned on AdventureWorks; a held-out run on BIRD has not been done.

### 5.5 `graph/ingest.py` — Cameron

```python
def create_constraints_and_indexes() -> None
    # UNIQUE on :Table(uid), :Column(uid); indexes on :Table(name), :Column(name), db_id
def clear_database(db_id: str) -> None
    # deletes only this db_id, in batches
def ingest_metadata(meta: DatabaseMetadata, session=None, rebuild: bool = False,
                    batch_size: int = 500) -> IngestReport
    # Schema/Table/Column nodes, CONTAINS, HAS_COLUMN, REFERENCES; idempotent (UNWIND/MERGE)
def apply_weights(db_id: str) -> None
    # sets weight_forward and weight_reverse on every REFERENCES edge
def verify_against_metadata(meta: DatabaseMetadata) -> list[str]
    # returns discrepancies (missing tables, etc.); empty list = OK
```

Every node carries `db_id`. `REFERENCES` edges carry `source`, `confidence`, `cardinality`, `constraint_name`, `evidence_json` (JSON text), `weight_forward`, `weight_reverse`, `from_table`, `to_table`, `db_id`. CLI: `python -m schemagraph.graph.ingest --db X [--meta PATH] [--rebuild]`. Functions take an optional `session` (a Neo4j session, or a fake in tests) and otherwise open one from `graph/driver.py`.

### 5.6 `graph/weights.py` — Cameron (pure functions, unit tested)

```python
def edge_weight(confidence: float, direction: Literal["forward", "reverse"]) -> float
    # 1 + 5 * (1 - confidence) + 1 if direction == "reverse"; rounded to 6 places
def edge_weights(confidence: float, cardinality: str) -> tuple[float, float]
    # (forward, reverse); for 1:1 the reverse equals the forward
```

**(changed)** `source` is no longer a parameter because `confidence` already encodes it (declared = 1.0), and the penalties are module constants, so `settings` is gone. `"forward"` is `many_to_one` and `"reverse"` is `one_to_many`. A declared edge costs 1.0 forward and 2.0 reverse. **(open: Cameron to confirm the signature.)**

### 5.7 `graph/enrich.py` — Cameron

```python
def generate_missing_descriptions(db_id: str) -> int
    # LLM writes descriptions from names + sample values; sets description_source="generated"
    # NOT YET WRITTEN
def embed_nodes(db_id: str, embedder: Embedder | None = None, session=None,
                include_generated: bool = True, batch_size: int = 500) -> EmbedReport
    # EmbedReport(tables, columns); stores vectors in the `embedding` property
```

`graph/vector.py` holds the pieces `embed_nodes` and `retrieve` share **(changed: `create_vector_indexes` moved here and takes the dimension)**:

```python
class Embedder(Protocol):                 # same two methods as LangChain Embeddings
    def embed_query(self, text: str) -> list[float]
    def embed_documents(self, texts: list[str]) -> list[list[float]]
class OpenAICompatibleEmbedder            # POST {base_url}/embeddings; batches; retries 429/5xx; checks dimensions
def get_embedder() -> Embedder            # from config.yml `embedding:`
def table_text(table, include_generated=True) -> str     # schema.name (humanized) + description
def column_text(table, column, include_generated=True) -> str   # schema.table.column (type): description. Examples: up to 3
def create_vector_indexes(dimensions: int) -> None       # table_embedding, column_embedding; cosine; IF NOT EXISTS
def drop_vector_indexes() -> None
```

`include_generated=False` supports the ablation without generated descriptions: descriptions whose `description_source` is `"generated"` are left out of the embedded text. Vector indexes need Neo4j 5.13 or later. An existing index keeps its dimension, so changing the embedding model means dropping the indexes and re-embedding every database.

### 5.8 `graph/queries.py` — Cameron

```python
def get_tables(db_id: str, uids: list[str]) -> list[TableDetail]   # table + columns, for DDL
def get_all_tables(db_id: str) -> list[TableDetail]                # full-schema mode
def list_tables(db_id: str) -> list[TableSummary]                  # /api/schema
def get_table_detail(db_id: str, uid: str) -> TableWithRelationships
def get_reference_edges(db_id: str, include_inferred: bool = True) -> list[ReferenceEdge]
    # every edge with column and table ends and both weights; the planner's input
```

`get_tables` keeps the order of `uids` and skips unknown UIDs with a warning, so a stale retrieval result cannot break prompt building. `get_table_detail` raises `TableNotFound`; the API should map it to a client error (the spec's 400 for unknown input).

### 5.9 `graph/retrieve.py` — Cameron

```python
def retrieve_seed_tables(db_id: str, question: str, k: int, *, embedder=None, session=None,
                         candidates: int = 100, min_score: float | None = None) -> list[SeedTable]
    # vector search over tables and columns; column hits roll up to their table (max score)
def rollup(table_hits, column_hits, k) -> list[SeedTable]      # pure
```

The question is embedded once. The two vector indexes hold every `db_id`, so each search over-fetches (`candidates` tables, three times that for columns) and keeps only hits with `node.db_id = $db_id`. A table's score is the best of its own and its columns'; `matched_on` is `"table"` when the table's own score is at least the best column score, otherwise `"column:<name>"`. Results are sorted by score then UID, scores rounded to 4 places. Fewer than `k` come back if the database has fewer; an empty question returns `[]`.

### 5.10 `graph/planner.py` — Cameron

```python
def plan_joins(db_id: str, seeds: list[SeedTable], include_inferred: bool = True, weighted: bool = True,
               *, max_bridge_tables: int = 4, algorithm: str | None = None, session=None) -> JoinPlan
def plan_from_edges(db_id, seeds, edges, *, include_inferred=True, weighted=True,
                    max_bridge_tables=4, algorithm=None, max_alternatives=2) -> JoinPlan
    # the same search on in-memory edges; plan_joins reads edges with queries.get_reference_edges
```

**(changed: no GDS.)** Internally: collapse column edges to table-level arcs (forward for many-to-one; reverse, one-to-many, only for N:1 edges; a composite foreign key is one link) → exact directed Steiner tree by Dreyfus–Wagner over all seed subsets for up to 7 seeds (two seeds is a shortest path; one DP answers every root), or NetworkX's Mehlhorn approximation on undirected minimum-direction costs for more seeds or `algorithm="networkx"` → check `max_bridge_tables`, dropping the lowest-score seed (listed in `dropped_seeds`) until the plan fits → map table links back to column pairs → find up to 2 runner-up paths by banning each bridge in turn, keeping only those with distinct table sets, for `alternatives`. `include_inferred=False` and `weighted=False` support the ablations. **Never raises for "no path"**; returns `status="partial"` or `"none"` instead.

### 5.11 `pipeline/context.py` — Mohammed

```python
def build_context(db_id: str, question: str, mode: Mode, top_k: int) -> tuple[SchemaContext, list[SeedTable]]
def render_ddl(tables: list[TableDetail]) -> str      # CREATE TABLE text with PK/FK and comments
def render_joins(plan: JoinPlan) -> str
```

In `graph` mode, if `plan.status == "none"`, fall back to retrieval context and add a note.

### 5.12 `pipeline/examples.py` — Mohammed

```python
def retrieve_examples(db_id: str, question: str, k: int = 3) -> list[FewShotExample]
    # from fewshot_examples.jsonl only
```

### 5.13 `pipeline/generate.py` — Mohammed

```python
def build_prompt(question: str, context: SchemaContext, examples: list[FewShotExample],
                 previous_attempts: list[Attempt]) -> list[BaseMessage]
def generate_sql(question: str, context: SchemaContext, examples: list[FewShotExample],
                 previous_attempts: list[Attempt]) -> GenerationResult
    # GenerationResult(sql, raw_output, input_tokens, output_tokens)
def extract_sql(llm_output: str) -> str | None
```

The prompt template is identical across modes; only the context sections differ.

### 5.14 `pipeline/validate.py` — Mohammed

```python
def is_single_select(sql: str) -> bool        # sqlglot parse; rejects writes and multiple statements
def validate(db_id: str, sql: str) -> ExplainResult
```

### 5.15 `pipeline/explain.py` — Mohammed

```python
def build_explanation(question: str, sql: str, context: SchemaContext,
                      seeds: list[SeedTable]) -> Explanation
```

Tables and join path are built from the join plan with templates (no LLM needed). Only `summary` uses the LLM. `interpretation` is filled when `alternatives` is non-empty.

### 5.16 `pipeline/pipeline.py` — Mohammed

```python
def answer_question(request: AskRequest, explain: bool = True) -> AskResponse
```

The single entry point. **Both the API and the evaluation harness call this function**, so evaluation measures exactly what users get. `explain=False` lets evaluation skip the explanation's LLM call.

Flow: `build_context` → `retrieve_examples` → loop up to `max_attempts` { `generate_sql` → `is_single_select` → `validate` } → `run_readonly_query` → `build_explanation` → assemble `AskResponse` with timings.

### 5.17 `api/main.py` — Mohammed

Thin routes only: validate input, call `answer_question` / planner / queries, map exceptions to HTTP codes. CORS enabled for `localhost:5173`.

### 5.18 `eval/compare.py` — Emmanuel (unit tested)

```python
def results_match(gold: QueryResult, predicted: QueryResult, order_matters: bool) -> bool
    # compares as multisets of rows unless order matters; normalizes numbers and NULLs
```

### 5.19 `eval/schema_metrics.py` — Emmanuel (unit tested)

```python
def tables_in_sql(sql: str, db_id: str) -> set[str]                 # sqlglot, postgres dialect
def schema_recall(gold_tables: set[str], context_tables: set[str]) -> float
def bridge_tables_needed(gold_tables: set[str], seed_tables: set[str]) -> set[str]
```

### 5.20 `eval/run_eval.py` — Emmanuel

```python
def load_questions(path: Path, split: str = "test") -> list[Question]
def run_eval(question_files: list[Path], modes: list[Mode], out_path: Path,
             run_id: str, planner_options: dict | None = None) -> None
def summarize(results_path: Path) -> pd.DataFrame     # accuracy etc. by mode × tier
```

Gold SQL is executed with `run_readonly_query` and cached, so each gold query runs once per run.

### 5.21 `scripts/build_graph.py`

```
python scripts/build_graph.py --db adventureworks [--skip-inference] [--skip-enrichment]
```

Runs `extract_metadata` → `infer_relationships` → `save_metadata` → `ingest_metadata` → `apply_weights` → `generate_missing_descriptions` → `create_vector_indexes` → `embed_nodes` → `verify_against_metadata`.

**Written** (`schemagraph/graph/build.py`, run through `scripts/build_graph.py`). It also takes `--all`, repeated `--db`, `--rebuild` and `--skip-extraction`. `generate_missing_descriptions` is not a step yet. Neo4j, the embedding config and the embedding size are checked once before any database is touched (exit 2); after that one database failing does not stop the others (exit 1 if any failed or `verify_against_metadata` reports differences). Embedding is saved per chunk, tables first, and resumes: nodes are fingerprinted by text and model, and a rerun embeds only missing or changed ones (`--reembed` forces all). Rate limiting (HTTP 429) waits up to `embedding.max_wait_s`, then stops with a partial result and exit code 3.

### 5.22 Frontend — Davin

**`src/api/client.ts`**

```ts
getHealth(): Promise<Health>
getDatabases(): Promise<DatabaseInfo[]>
ask(req: AskRequest): Promise<AskResponse>
plan(req: AskRequest): Promise<PlanResponse>
listTables(dbId: string): Promise<TableSummary[]>
getTable(dbId: string, uid: string): Promise<TableWithRelationships>
```

**Components:** `DatabaseSelector`, `QuestionInput`, `ResultTable`, `SqlView`, `TablesUsed` (seed vs. bridge badges), `JoinPathView`, `ExplanationPanel`, `WarningsView`, `AttemptsView` (collapsed by default), and a `ModeSelector` shown only in a developer view.

Build against `src/mocks/ask_response.json` until the backend is ready, with a `VITE_USE_MOCKS=true` switch in `client.ts`.

---

## 6. Working in parallel

### 6.1 What each person can build before anyone else is finished

| Owner | Starts from |
|---|---|
| Emmanuel | PostgreSQL alone. Writes extraction and the eval harness (gold queries run without the pipeline). |
| Cameron | `contracts/examples/metadata.adventureworks.sample.json` (a small hand-written file covering ~6 tables) until real extraction output exists. |
| Mohammed | A stub `retrieve_seed_tables` returning fixed tables and a stub `plan_joins` returning `join_plan.sample.json`. Full-schema mode needs only `get_all_tables`, which can also be stubbed from the sample metadata. |
| Davin | `ask_response.sample.json` and the mock switch. |

### 6.2 Integration milestones

1. **End of week 1:** full-schema mode answers a question end to end through the API and frontend (stubs allowed elsewhere).
2. **End of week 2:** real extraction → real graph → retrieval mode working.
3. **End of week 3:** graph mode working; `run_eval.py` produces a results file for all three modes.

### 6.3 Repository rules

- `main` always runs. Work on branches; merge through pull requests with one reviewer.
- Changes to `models.py`, `config.py`, or `contracts/` need review from everyone affected.
- Every new contract change updates the matching sample in `contracts/examples/`.
- Never commit `.env` or API keys.

---

## 7. Safety requirements

- All LLM-generated SQL runs through `run_readonly_query`, using a PostgreSQL role with `SELECT` only.
- `statement_timeout` is set on every read-only connection.
- `is_single_select` must pass before `explain` is even called.
- The admin connection is used only by extraction, inference, and data loading.
- Two PostgreSQL roles: `sg_admin` (extraction, inference, loading) and `sg_readonly` (`SELECT` only; created by `scripts/create_roles.sql`, which must be re-run after loading a database so the grants cover its tables). Passwords come from environment variables named in `config.yml`.
- Nothing else executes model-written SQL: it passes `is_single_select`, then `run_readonly_query`.

---

## 8. Implementation status and decisions

### 8.1 Build status

| Module | Owner | Status |
|---|---|---|
| `config.py`, `models.py` | shared | Built, tested |
| `db/postgres.py` | Emmanuel | Built, tested |
| `extraction/extract.py`, `infer.py`, `build_metadata.py` | Emmanuel | Built, tested (inference tuned on AdventureWorks only) |
| `eval/infer_recovery.py` | Emmanuel | Built |
| `eval/compare.py`, `schema_metrics.py`, `run_eval.py`, `eval/questions/*` | Emmanuel | Not started |
| `graph/weights.py`, `driver.py`, `cypher.py`, `ingest.py`, `queries.py`, `planner.py` | Cameron | Written and tested with fakes; Cypher not yet run on a real Neo4j |
| `graph/vector.py`, `retrieve.py`, `enrich.embed_nodes` | Cameron | Written and tested with fakes and a local HTTP server; Cypher not yet run on a real Neo4j |
| `graph/enrich.generate_missing_descriptions` | Cameron | Not started |
| `scripts/build_graph.py` | Cameron / Emmanuel | Written and tested with fakes; not yet run on a real Neo4j |
| `export_contracts.py` | Cameron / Emmanuel | Not started |
| `pipeline/*`, `api/main.py` | Mohammed | Not started |
| `frontend/` | Davin | Not started |
| `contracts/examples/`: metadata and join plan samples | | Done. `ask_response.sample.json` still to write |

### 8.2 Changes from the first draft

1. **Config is `config.yml`, not `.env`** (5.1). Two Postgres roles, optional `neo4j` and `embedding` sections.
2. **No GDS.** The planner is an exact Steiner solver with a NetworkX fallback (5.10). `exact_steiner` is a new `algorithm` value; `gds_steiner` is reserved.
3. **`edge_weight(confidence, direction)`** replaces the four-argument form (5.6).
4. **Embeddings** go through an `Embedder` interface and any OpenAI-style endpoint; `create_vector_indexes` takes the dimension (5.7).
5. **`db_id` format** is fixed (`^[a-z0-9_]+$`), Schema uids are `{db_id}:{schema}`, and UIDs are never parsed (2.1).
6. **1:1 edges** cost the same in both directions; inferred confidence is capped at 0.95 (2.2, 3.1).
7. Several functions gained optional `session`, `rebuild`, `sample_n` and similar parameters. These are additive and do not break callers written to the first draft.

### 8.3 Open decisions

- Confirm with Cameron: the Schema uid format, the 1:1 reverse weight, the 0.95 inferred cap, the `edge_weight` signature, the `exact_steiner` contract value, and listing unreachable seeds without including them in `tables`.
- The embedding model and `dimensions` (set in `config.yml`; they fix the vector index).
- Whether planner constants (`CONFIDENCE_PENALTY`, `FAN_OUT_PENALTY`, `MAX_BRIDGE_TABLES`) move into config.
- A held-out BIRD run to check that inference and planning are not tuned only to AdventureWorks.

### 8.4 Testing

Unit tests use pure builders and injected fakes (a fake Neo4j session, a fake vector store, a local HTTP server for the embedder) and need no services. Integration tests are gated by environment variables: `SG_TEST_PG_DSN`, `SG_TEST_ADVENTUREWORKS=1`, `SG_TEST_NEO4J=1` (with `NEO4J_PASSWORD`), under the markers `integration`, `adventureworks` and `neo4j`. New logic is mutation-checked: break the code and confirm a test fails.
