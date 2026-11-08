# SchemaGraph

Text-to-SQL for PostgreSQL. A question goes in; a checked, read-only SQL query and its result come out.

```
Postgres ──extract/infer──▶ metadata JSON ──ingest──▶ Neo4j graph ──▶ retrieve ──▶ join planner ──▶ LLM ──▶ SQL ──▶ result
 (catalog)    (Emmanuel)     (the contract)   (Cameron)   (tables,       (tables?)    (join path)   (Mohammed)       (read-only role)
                                                          columns,
                                                          joins)
```

The metadata JSON is the hand-off between the extraction side and the graph side. Everything
downstream reads that file's shape (`schemagraph/models.py`), so changes to it need everyone's review.

## Who owns what

| Person | Slice | Where | Status |
| --- | --- | --- | --- |
| **Emmanuel** | Postgres access, metadata extraction, relationship inference, evaluation harness | `schemagraph/db/`, `schemagraph/extraction/`, `schemagraph/eval/`, `scripts/` | Postgres access, extraction and inference built and tested. Evaluation code written and tested with fakes: `compare.py`, `schema_metrics.py`, `run_eval.py` (it needs `answer_question`, and the question files, to run for real). `infer_recovery.py` also exists |
| **Cameron** | Graph: weights, ingest, retrieval, join planner, graph queries | `schemagraph/graph/` | `weights`, `driver`, `ingest`, `queries`, `planner` written; Cypher needs a first run on a real Neo4j. Retrieval and embeddings written (not yet run on a real Neo4j); description generation not started |
| **Mohammed** | LLM pipeline and FastAPI | `schemagraph/pipeline/`, `schemagraph/api/` | Context building, few-shot selection, OpenAI-compatible generation, SQL validation, explanation, and `answer_question` implemented; FastAPI is not started |
| **Davin** | React frontend | `frontend/` | Not started |

The full design is in `SPEC.md`. Section numbers quoted in the code (3 for contracts, 5.6 for edge
weights, 7 for the two Postgres roles) refer to it.

### If you are Cameron (graph)

- **Input:** a `DatabaseMetadata` file. Start from `contracts/examples/metadata.adventureworks.sample.json`
  (7 tables, 93 columns, 5 relationships, one of them inferred). No database is needed to work with it.
- **Already there:** `graph/weights.py` (edge cost, SPEC 5.6), `graph/driver.py` (one shared Neo4j driver),
  `graph/ingest.py` (constraints, ingest, `apply_weights`, `verify_against_metadata`, `clear_database`).
- **Graph shape:** `(:Schema)-[:CONTAINS]->(:Table)-[:HAS_COLUMN]->(:Column)` and
  `(:Column)-[:REFERENCES]->(:Column)`, always from the FK column to the referenced column. Every
  `REFERENCES` edge carries `source`, `confidence`, `cardinality`, `weight_forward`, `weight_reverse`,
  `evidence_json`, `from_table` and `to_table`.
- **Decisions to confirm:** Schema uid is `{db_id}:{schema}`; a 1:1 edge costs the same in both directions;
  inferred confidence is capped at 0.95, so an inferred edge always costs more than a declared one.
- **First thing to do:** run `tests/test_ingest_neo4j.py` against your Neo4j (see Testing). The Cypher has only
  been tested against a fake session so far.
- **Weights:** forward = following the arrow (many-to-one), reverse = against it (one-to-many, can multiply rows).
  Declared forward is 1.0, declared reverse is 2.0. Note the signature differs from SPEC 5.6:
  `edge_weight(confidence, "forward" | "reverse")` and `edge_weights(confidence, cardinality)` instead of
  `edge_weight(source, confidence, traversal, settings)`. The planner's `traversal` values
  `"many_to_one"` / `"one_to_many"` map to `"forward"` / `"reverse"`. Agree which form you want and change it once.
- **Also written:** `graph/queries.py` (`get_tables`, `get_all_tables`, `list_tables`, `get_table_detail`, plus
  `get_reference_edges` for the planner, which returns every edge with both column and table ends and its weights)
  and `graph/cypher.py` (session helpers shared by ingest and queries).
- **Also written:** `graph/planner.py` (`plan_joins`, SPEC 5.10). It finds the cheapest join tree with an exact
  directed Steiner solver (up to 7 seeds; two seeds is shortest path) and falls back to NetworkX's approximation for more
  seeds or when asked (`algorithm="networkx"`). It supports the two ablations (`include_inferred=False`,
  `weighted=False`), drops the weakest seed when a plan needs more than 4 bridge tables, and returns runner-up
  `alternatives`. It never raises for "no path". It reads edges through `queries.get_reference_edges`, so the search
  itself runs and is tested without Neo4j. GDS is not used; `gds_steiner` stays in the contract for later.
- **Still to build:** `enrich.generate_missing_descriptions`. `scripts/build_graph.py` runs the rest of the flow
  (extract, infer, save, ingest, weights, vector indexes, embeddings, verify).

### If you are Mohammed (pipeline and API)

- **Read-only SQL:** everything the model writes must run through `schemagraph.db.postgres.run_readonly_query`.
  It uses the `sg_readonly` login, a statement timeout, and read-only transactions, and returns at most
  `row_limit` rows with a truncation flag. `explain()` is there for checking a query without running it.
  Do not open your own Postgres connection for LLM SQL.
- **Databases:** `list_databases()` gives what the read-only role can connect to. You will probably want to
  intersect it with the databases that have a graph.
- **Contracts:** the result types (`GenerationResult`, `ExplainResult`, `TableDetail`, `TableSummary`, ...) are in
  `schemagraph/models.py`. Dump with `by_alias=True`: two fields are aliased because their names clash with
  Python (`schema`, `from`).
- **Config:** add your section to `config.py` (`llm`, `pipeline`) the way `neo4j` was added: a Pydantic model,
  an optional field on `AppConfig`, secrets by environment variable name. Never put a key in `config.yml`.
- **Chat model:** `llm` uses an OpenAI-compatible `/chat/completions` endpoint. Set the model and `api_key_env` in
  `config.yml`; the key itself belongs only in the environment. Local compatible servers can set `api_key_env: null`.
- **Pipeline defaults:** `pipeline` supports `top_k`, `max_attempts`, `row_limit`, `examples_k`, `fewshot_path`, and (off by default) `expand_seeds` / `expand_hops`.
  Few-shot files must contain development examples only.

Example additions to `config.yml`:

```yaml
llm:
  provider: openai_compatible
  base_url: https://api.openai.com/v1
  model: your-chat-model
  api_key_env: OPENAI_API_KEY
  timeout_s: 60
  temperature: 0
  max_tokens: 2048
pipeline:
  top_k: 5
  max_attempts: 3
  row_limit: 100
  examples_k: 3
  fewshot_path: schemagraph/eval/questions/fewshot_examples.jsonl
```

### If you are Davin (frontend)

- **API shapes** come from the Pydantic models in `schemagraph/models.py`; those are the JSON your app
  receives. `contracts/examples/` holds real example data, and more will be added there.
- **Name display:** UIDs look like `db:schema.table.column`. Show `models.display_name(uid)`, and never split a
  UID yourself, because table and column names can contain dots.
- **Inferred relationships** have `source: "inferred"`, a `confidence` below 1.0 and an `evidence` object
  (`name_match`, `type_compatible`, `value_overlap`). Declared ones have `confidence: 1.0` and no evidence.
  The UI should make the difference visible.

### If you are Emmanuel (extraction)

Details are in "What each folder does" below. The evaluation code (SPEC 5.18 to 5.20) is written; see "Running the
evaluation" under Testing. Open items: the AdventureWorks question file and the few-shot file (`eval/questions/`), the held-out inference
test on BIRD, `.env` loading for passwords, and the database-wrapper abstraction (deferred on purpose).

## Setup

Requirements: Python 3.11+, Docker, and `psql`/`pg_dump` version 15 or newer (`create_roles.sql` uses `\getenv`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Databases

`docker-compose.yml` runs Postgres 17 and Neo4j 5 (community), both bound to `127.0.0.1` only.
Put the passwords in a `.env` file next to it (git-ignored; never commit it):

```
DB_PASSWORD=...          # Postgres superuser "app"
APP_DB_PASSWORD=...
NEO4J_PASSWORD=...       # at least 8 characters; read once, when the volume is first created
```

```bash
docker compose up -d
```

### Configuration

```bash
cp config.example.yml config.yml     # not committed
export SCHEMAGRAPH_CONFIG=$PWD/config.yml
export PG_ADMIN_PASSWORD=...  PG_READONLY_PASSWORD=...  NEO4J_PASSWORD=...
```

`config.yml` names environment variables for each password (`password_env`); the passwords themselves
never go in the file. The two Postgres logins exist for a reason:

- `sg_admin` reads the catalog and samples data (extraction, inference, loading). It never runs LLM SQL.
- `sg_readonly` has SELECT only, plus a timeout and read-only transactions. It runs every LLM-generated query,
  so a bad or malicious query cannot change data or schema. The database enforces this, not a check on the SQL text.

### Data sources

There are two databases. Neither is stored in this repository; each load script fetches or reads them.

| Source | What it is | Where to get it | Used for |
| --- | --- | --- | --- |
| **AdventureWorks** | Microsoft's sample company database: 68 tables in 5 schemas (`humanresources`, `person`, `production`, `purchasing`, `sales`). Nearly every foreign key is declared. | Two parts. (1) The Postgres port: clone <https://github.com/lorint/AdventureWorks-for-Postgres>. (2) Microsoft's CSV data, which `load_adventureworks.sh` downloads for you from <https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks-oltp-install-script.zip> (about 17 MB). | The main evaluation database: cross-schema joins, hand-written questions (about 60, split 50 test / 10 dev). |
| **BIRD mini-dev** | A public Text-to-SQL benchmark: 500 questions over about a dozen small databases, each with an `evidence` hint. Ships with a PostgreSQL dump and PostgreSQL-flavoured gold queries. | The `bird-bench/mini_dev` git repo (<https://github.com/bird-bench/mini_dev>) holds code only, not the data. The data is a separate download, linked from its README: the Aliyun zip <https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip> (the server is in Beijing and often stalls from outside China), or the "complete package" on Google Drive (<https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view>). The question files are also on Hugging Face (<https://huggingface.co/datasets/birdsql/bird_mini_dev>, split `mini_dev_pg`); I have not confirmed that it includes the database dump. | A held-out test for relationship inference and for evaluation on databases the rules were not tuned on. |

Inside the BIRD zip, the files we use are `MINIDEV_postgresql/BIRD_dev.sql` (the dump) and
`mini_dev_postgresql.json` (the questions). BIRD's own instructions load the whole dump into one database
called `BIRD`; SchemaGraph needs one database per `db_id`, which `load_bird.sh` arranges. Check each dataset's
licence terms before redistributing it or committing any of it to the repo.

### Load sample data

```bash
export PGHOST=127.0.0.1 PGPORT=5432 PGUSER=app PGPASSWORD=$DB_PASSWORD   # a superuser

# AdventureWorks. Needs git, Ruby (to convert the CSVs once) and a clone of the lorint repo:
git clone https://github.com/lorint/AdventureWorks-for-Postgres adventureworks-for-postgres
scripts/load_adventureworks.sh --repo ./adventureworks-for-postgres

# BIRD mini-dev. Download minidev.zip first (table above), then:
scripts/load_bird.sh --zip ~/Downloads/minidev.zip
# if the dump layout is not recognised the script stops and prints the table list; send that to Emmanuel

# Already have every BIRD table in one database (`bird_staging`, all in schema public)? Skip the download and
# split it into one database per db_id using scripts/bird_tables.tsv:
PG_ADMIN_PASSWORD=... PG_READONLY_PASSWORD=... scripts/load_bird.sh --from-staging --force
```

The BIRD mini-dev dump puts all 11 databases' tables into a single schema. `load_bird.sh` splits them with
`scripts/bird_tables.tsv` (one `db_id<TAB>table` line per table). Before splitting it compares that file with the
real tables and stops (exit 2, nothing changed) if they differ, so edit the file rather than the script. After the
split it checks each table's row count against staging. `--force` drops and recreates the per-db databases;
`bird_staging` is kept until you drop it.

Both scripts create lowercase UTF8 databases, run `scripts/create_roles.sql`, run `ANALYZE` and check that
`sg_readonly` can read but not write. A database loaded any other way needs
`psql -d <db> -f scripts/create_roles.sql` as a superuser, or `sg_admin` gets "permission denied" on its tables.

### Build the metadata and the graph

```bash
python -m schemagraph.extraction.build_metadata --db adventureworks     # writes data/metadata/adventureworks.json
python -m schemagraph.graph.ingest --db adventureworks --rebuild        # loads it into Neo4j and verifies
```

`build_metadata` accepts `--all`, repeated `--db`, and `--skip-inference`.

To do everything in one command (extract, infer, ingest, weights, vector indexes, embeddings, verify):

```bash
python scripts/build_graph.py --db adventureworks --rebuild
python scripts/build_graph.py --all --skip-enrichment             # no embedding model needed
python scripts/build_graph.py --db adventureworks --skip-extraction   # reuse data/metadata/adventureworks.json
```

Before touching any database it checks that Neo4j answers and, unless `--skip-enrichment`, that the configured embedding
model is reachable and returns vectors of the size `config.yml` says (a wrong size would leave the index silently empty).
Those failures exit 2 with nothing changed. After that, one database failing does not stop the others; the exit code is 1 if
any failed or the graph does not match its metadata, and 3 if embedding was cut short by rate limiting (below).
`--rebuild` clears each db_id's nodes first, which also removes its embeddings, so use it with enrichment on.
If you change a model's `dimensions`, call `vector.drop_vector_indexes(name=...)` first, because an existing index keeps
its old size. LLM-written descriptions are not a step yet.

**Rate limits and resuming (free-tier API keys).** Embeddings are saved as they are made: tables first, then columns, one
request's worth of texts at a time. A rate-limited request (HTTP 429) waits as long as the service asks (`Retry-After`, or
Google's `retryDelay`), at least 5, 15, 30, then 60 seconds, until `max_wait_s` (default 120) is used up. Then embedding stops
cleanly: the run prints `PART ... embedded 7/7 tables and 40/93 columns`, keeps everything saved, and exits 3. Rerun the same
command **without `--rebuild`** (it would clear the vectors) and only what is missing is embedded; add `--skip-extraction` to
reuse the metadata file. Every stored vector carries a fingerprint of its text and model, so a rerun also re-embeds nodes whose
description changed, or whose model, provider or dimensions changed under the same `name`. `--reembed` redoes everything.
Vectors written before fingerprints existed are replaced once. In `config.yml` under `embedding:`:
`max_wait_s` (a longer budget rides out per-minute limits; a daily quota will not clear in any budget, so the run stops) and
`requests_per_minute` (spaces requests out so the limit is not hit). Gemini sends 100 texts per request unless `batch_size` says
otherwise, which matters on tiers limited by requests per minute. Retrieval skips nodes without a vector, so do not rely on a
partial graph for answers; the coverage numbers in the summary say how far it got.

**Embedding quota.** Every `retrieval` and `graph` call embeds its question, and free tiers count each call against a daily
limit, so a sweep over four settings would embed each question four times. With `embedding.query_cache` set (it is in
`config.yml`), a question is embedded once and reused from `data/cache/query_embeddings.sqlite` by every mode, setting and
rerun; the key includes the provider, model and dimensions, so changing the model starts fresh. `run_eval` and `retrieval_eval`
also embed all their questions in advance, `batch_size` per request (one request for up to 100 questions), before the first
model call; if the quota is already gone, `run_eval` says so and exits with status 3 before running anything. If the service still refuses
three questions in a row (quota used up), `run_eval` stops instead of failing the rest, prints `STOPPED: ...`, exits with
status 3 and keeps every finished row: rerun the same command later (a daily quota resets overnight, Pacific time for Google) and it
carries on from where it stopped. `--modes full` needs no embedding at all.

### Smoke test the whole path

After building a graph, push a few questions through the real services before any long run:

```bash
python scripts/smoke_test.py --db adventureworks                       # five built-in questions, all three modes
python scripts/smoke_test.py --db adventureworks --question "How many employees are there?" --mode graph
python scripts/smoke_test.py --db card_games --questions-file schemagraph/eval/questions/bird_subset.jsonl --n 5
python scripts/smoke_test.py --db adventureworks --no-llm              # retrieval and join planning only
```

For each question and mode it prints what retrieval found (with scores), the join plan, the tables the model was given,
and, with an `llm:` section in `config.yml`, the SQL, whether it ran, and its cost. With a questions file it also says
whether the result matches the gold result and what share of the gold tables the model was given. It stores nothing.
Exit code 1 means some question hit an error (shown on its line), 2 means it could not start.

### Score retrieval on its own

The smoke test shows a few questions in detail; this scores all of them, with no LLM and no Postgres:

```bash
python scripts/retrieval_eval.py --db adventureworks --questions-file schemagraph/eval/questions/adventureworks.jsonl \
    --label gemini-1536 --out runs/gemini.json
python scripts/retrieval_eval.py --compare runs/openai.json runs/gemini.json     # side by side
```

It embeds each question once and reports (1) recall@k: how many of the tables the gold SQL reads are among the k
best-scoring tables, (2) the mean score at ranks 1, 5 and 20, which shows whether scores separate relevant tables from
noise, and (3) for each seed strategy (`top_k`, with or without adding up to N tables the seeds have foreign keys to),
the planner's context as the model would see it: recall and the average number of tables. Use it to choose `top_k`
and to compare embedding models and embedding texts: build one graph (or one named vector set, `embedding.name`)
per variant, run once per variant with `SCHEMAGRAPH_CONFIG=<config>` and `--out`, then `--compare`. Runs are
comparable only on the same question file and split.
`--no-graph` skips the planning part.

Two ways to bring in tables a question never names (the person behind a salesperson, the product under a category):

- `embedding.include_relationships: true` embeds each table with its foreign-key neighbours, so a link table matches
  questions about what it links. It changes the embedded text; give the config a new `embedding.name` and run
  `python scripts/build_graph.py --db <db> --skip-extraction` to embed under it (the plain vectors are kept).
- `pipeline.expand_seeds: N` (graph mode) adds up to N tables that the retrieved tables reference, `expand_hops` links
  away, before join planning. The retrieval script's `+N (H-hop)` rows measure it before you turn it on; the
  explanation marks an added table "Added because X has a foreign key to it".

## API reference

The backend is a FastAPI app (`schemagraph/api/main.py`, Mohammed). It serves OpenAPI docs at `/docs` and the
schema at `/openapi.json`; the frontend generates `types.ts` from that. CORS allows `http://localhost:5173`.
`SPEC.md` section 3 and 4 is the source of truth; this is the short version. Where the two differ, fix this file.

### Running it

```bash
pip install -e '.[api]'
uvicorn schemagraph.api.main:app --reload        # http://localhost:8000/docs
```

There is one code path: every `/api/ask` call runs `answer_question`, the function the evaluation harness uses, so
the evaluation measures what users get (there is no demo or sample mode). Status codes follow `SPEC.md` 3.5: `200` for
every completed run, including failures reported in `status`; `400` for bad input (empty question, unknown `db_id`,
invalid `mode` or `options`; never FastAPI's default `422`); `404` for a table that is not in the graph; `503` when
PostgreSQL, Neo4j, the embedding service or the LLM cannot be reached. A `503` names the failing dependency and nothing
else (no URL, key or trace; those go to the server log), and carries `Retry-After` when the embedding service is rate
limited. `GET /api/health` reports `ok`, `error` or (LLM only) `not_configured` per dependency; add `?deep=true` to make a
real one-line LLM call (otherwise it only checks that an `llm` section and its key exist, so polling it costs nothing).
`/api/databases` lists databases that are both in PostgreSQL and have a graph built.

### Endpoints

| Method | Path | Purpose | Response |
| --- | --- | --- | --- |
| `GET` | `/api/health` | Checks PostgreSQL, Neo4j and the LLM | `{"postgres": "ok", "neo4j": "ok", "llm": "ok"}` |
| `GET` | `/api/databases` | Databases available to query (the selector) | `[{"db_id": "adventureworks", "table_count": 68, "schema_count": 5}]` |
| `POST` | `/api/ask` | Main endpoint: question to SQL, result and explanation | `AskResponse` |
| `POST` | `/api/plan` | Retrieval and join planning only, no LLM call. Same body as `/api/ask`. For debugging the graph on its own | `{"seed_tables": [...], "join_plan": {...}}` |
| `GET` | `/api/schema/{db_id}/tables` | Browse tables (lower priority; build after `/api/ask`) | `[{"uid", "display_name", "description", "column_count"}]` |
| `GET` | `/api/schema/{db_id}/tables/{table_uid}` | One table with columns and relationships (lower priority) | `{"table", "columns", "references", "referenced_by"}` |

Building the graph is not an endpoint. It runs from the command line.

### `POST /api/ask`

Request:

```json
{
  "question": "What is the total order value for each customer's name?",
  "db_id": "adventureworks",
  "mode": "graph",
  "options": { "top_k": 5, "max_attempts": 3, "row_limit": 200 }
}
```

`mode` is one of `full` (whole schema in the prompt), `retrieval` (top-k tables), or `graph` (retrieval plus a
join plan); it defaults to `graph`. `options` is optional and defaults come from `config.yml`.

Response (abridged):

```json
{
  "request_id": "6f1c2a0e-...",
  "question": "...", "db_id": "adventureworks", "mode": "graph",
  "status": "success",
  "sql": "SELECT p.firstname, p.lastname, SUM(soh.totaldue) ... GROUP BY p.firstname, p.lastname;",
  "result": { "columns": ["firstname", "lastname", "total_order_value"],
             "rows": [["Jon", "Yang", "8248.99"]], "row_count": 1, "truncated": false },
  "explanation": {
    "summary": "This adds up the total value of all orders for each customer...",
    "tables": [{ "uid": "adventureworks:sales.customer", "display_name": "sales.customer",
                 "role": "bridge", "reason": "Needed to connect orders to people." }],
    "join_path": [{ "from": "sales.salesorderheader.customerid", "to": "sales.customer.customerid",
                    "source": "declared", "text": "Each order belongs to one customer." }],
    "interpretation": "\"Person\" was interpreted as the customer, not the salesperson.",
    "warnings": []
  },
  "join_plan": { "...": "JoinPlan, or null outside graph mode" },
  "attempts": [{ "sql": "SELECT ...", "error": null }],
  "metrics": { "input_tokens": 1834, "output_tokens": 212, "attempt_count": 1,
              "latency_ms": { "retrieval": 120, "planning": 45, "generation": 2310, "validation": 18,
                              "execution": 32, "explanation": 1400, "total": 3925 } }
}
```

| Field | Values and notes |
| --- | --- |
| `status` | `success`; `invalid_sql` (failed validation after all attempts); `execution_error`; `no_schema` (nothing relevant retrieved); `unsafe_sql` (not a single SELECT) |
| `sql` | The final SQL, or `null` if none was produced |
| `result.rows` | Every value is a string or `null`, which avoids date and decimal problems in JSON. Capped at `row_limit`; `truncated` is true if more rows existed |
| `explanation.tables[].role` | `seed` (matched the question), `bridge` (needed to connect seeds), or `context` (full and retrieval modes) |
| `explanation.join_path[].source` | `declared` or `inferred`. Show the difference to the user |
| `attempts` | Every generated SQL in order, with the database error if it failed. Collapsed by default in the UI |

HTTP codes:

- `200` for every completed run, including failures reported in `status`, so the UI can still show the explanation and attempts.
- `400` for bad input: empty question, unknown `db_id`, invalid mode.
- `503` if PostgreSQL, Neo4j or the LLM is unreachable.
- Errors use FastAPI's default body, `{"detail": "..."}`.

### Join plan (`/api/plan`, and `join_plan` inside `/api/ask`)

`status` is `full` (all seeds connected), `partial` (some unreachable), `single_table` (one seed, no joins) or
`none` (no plan; the pipeline falls back to retrieval context). `algorithm` is `exact_steiner`,
`shortest_path` (two seeds), `networkx_steiner`, `gds_steiner` (reserved, not implemented) or `none`. Each entry in `joins` has `from_column`, `to_column` (always the
stored direction, FK column to referenced column, whichever way the plan walks it), `source`, `confidence`,
`cardinality` (`N:1` or `1:1`), `traversal` (`many_to_one` or `one_to_many`) and `weight`. The plan also lists
`seed_tables`, `bridge_tables`, `tables`, `total_cost`, `unreachable_seeds`, `dropped_seeds` and optional
`alternatives` (runner-up paths used to explain an ambiguous question). The planner never raises for "no path";
it returns `partial` or `none`.

### Internal interfaces each owner implements

Signatures are the contract; helpers are up to the owner. All are in SPEC section 5.

| Module | Owner | Main functions |
| --- | --- | --- |
| `db/postgres.py` | Emmanuel (done) | `list_databases()`, `run_readonly_query(db_id, sql, row_limit)`, `explain(db_id, sql)`, `run_admin_query(db_id, sql, params)` |
| `extraction/extract.py`, `infer.py` | Emmanuel (done) | `extract_metadata(db_id)`, `save_metadata`, `load_metadata`, `infer_relationships(db_id, meta)` |
| `eval/compare.py`, `schema_metrics.py`, `run_eval.py` | Emmanuel | `results_match(gold, predicted, order_matters)`, `schema_recall`, `bridge_tables_needed`, `run_eval(...)`, `summarize(...)` |
| `graph/ingest.py`, `weights.py` | Cameron (written) | `ingest_metadata`, `apply_weights`, `verify_against_metadata`, `clear_database`, `edge_weight` |
| `graph/queries.py` | Cameron (written) | `get_tables`, `get_all_tables`, `list_tables`, `get_table_detail`, `get_reference_edges` |
| `graph/planner.py` | Cameron (written) | `plan_joins(db_id, seeds, include_inferred, weighted)`, `plan_from_edges(...)` (same, on in-memory edges) |
| `graph/enrich.py`, `retrieve.py` | Cameron | `embed_nodes`, `retrieve_seed_tables(db_id, question, k)` |
| `pipeline/context.py`, `examples.py`, `generate.py`, `validate.py`, `explain.py` | Mohammed (written) | `build_context`, `retrieve_examples`, `generate_sql`, `is_single_select`, `validate`, `build_explanation` |
| `pipeline/pipeline.py` | Mohammed (written) | `answer_question(request, explain=True, *, planner_options=None)`: the single entry point |
| `api/main.py` | Mohammed | Thin routes: validate input, call the above, map errors to HTTP codes |
| `src/api/client.ts` | Davin | `getHealth`, `getDatabases`, `ask`, `plan`, `listTables`, `getTable` |

Two rules hold the pieces together:

1. **One entry point.** Both the API and the evaluation harness call `answer_question`, so evaluation measures
   exactly what users get.
2. **One way to run model SQL.** `is_single_select` (a parse that rejects writes and multiple statements) must
   pass, then the query runs through `run_readonly_query`. Nothing else executes generated SQL.

### File formats

- **Question files** (evaluation): JSONL, one question per line:
  `{"id": "aw-017", "db_id": "adventureworks", "question": "...", "gold_sql": "SELECT ...", "tier": 3, "split": "test", "order_matters": false, "evidence": null, "author": "cameron", "notes": ""}`.
  `tier` is 1 (single table), 2 (joins within a schema) or 3 (cross-schema, 3+ tables). `split` is `test` (about 50
  scored questions) or `dev` (about 10, only for tuning). `evidence` is BIRD's hint, `null` for AdventureWorks.
  `fewshot_examples.jsonl` has the same format and **must never contain test questions**.
- **Result files**: JSONL, one row per question per mode, with `correct`, `status`, `generated_sql`, `context_tables`,
  `schema_recall`, `bridge_tables_needed`, `bridge_tables_recovered`, `plan_status`, token and latency fields, and a `run_id`.
- **Examples to build against:** `contracts/examples/metadata.adventureworks.sample.json` and
  `join_plan.sample.json` exist now (the second is generated by the planner from the first, and a test keeps it
  in sync). `ask_response.sample.json` is a successful graph-mode answer to the same question; it embeds that join
  plan, and `tests/test_contract_examples.py` checks it against the models and the other samples. Still to write: the
  frontend mock (switched with `VITE_USE_MOCKS=true`), and samples for failure responses if the UI needs them.

## What each folder does

| Path | Purpose |
| --- | --- |
| `schemagraph/models.py` | Every shared data contract, plus the UID helpers. Change with care. |
| `schemagraph/config.py` | YAML config, the two Postgres roles, connection strings, Neo4j settings. |
| `schemagraph/db/` | Public PostgreSQL API: `list_databases`, `run_readonly_query`, `explain`, and `run_admin_query`; implementation is in `postgres.py`. |
| `schemagraph/sql_validation.py` | PostgreSQL-dialect check that accepts one read-only `SELECT` and rejects writes or multiple statements. |
| `schemagraph/extraction/sql/*.sql` | Catalog queries: tables, columns, primary keys, foreign keys, unique constraints. |
| `schemagraph/extraction/extract.py` | Turns those results into `DatabaseMetadata`; saves and loads the JSON. |
| `schemagraph/extraction/infer.py` | Finds relationships the database does not declare: name and type fit, confirmed by value overlap. |
| `schemagraph/extraction/build_metadata.py` | One command: extract, infer, save. |
| `schemagraph/eval/infer_recovery.py` | Hides declared foreign keys and measures how many inference recovers. |
| `schemagraph/graph/` | Weights, Neo4j driver, ingest, queries, planner. Cameron adds retrieval and enrichment here. |
| `scripts/` | Role creation and the two data loaders. |
| `contracts/examples/` | Example data files that teammates can build against. |
| `tests/` | Everything below. |

## Embedding models and retrieval

`graph/retrieve.py` finds the tables a question is about by vector search, so it needs an embedding model.
`graph/enrich.py` (`embed_nodes`) stores a vector on every table and column node, and `retrieve_seed_tables` searches them.
Set the model in the `embedding:` section of `config.yml` (commented examples are in the file).

| `provider` | Models | Key variable | Notes |
| --- | --- | --- | --- |
| `openai` | `text-embedding-3-large`, `-3-small`, or any OpenAI-style server (Ollama, vLLM) | `OPENAI_API_KEY` | `send_dimensions: true` makes OpenAI return shortened vectors. Use `api_key_env: null` for a local server |
| `voyage` (or `anthropic`) | `voyage-4`, `voyage-4-large`, `voyage-4-lite`, `voyage-code-4` | `VOYAGE_API_KEY` | Anthropic has no embedding model of its own and recommends Voyage AI, so `anthropic` is another name for `voyage`. Voyage embeds queries and documents differently; the code sets that for you |
| `gemini` | `gemini-embedding-2` (current, multimodal, stable), `gemini-embedding-001` (text only) | `GEMINI_API_KEY` | Default size is 3072; `send_dimensions: true` asks for fewer (Google recommends 768, 1536 or 3072). The two models tell queries from documents differently (`-001` a `taskType`, `-2` a text prefix) and the code does it for you. Their vectors are not compatible, so changing one for the other means re-embedding. The request shape follows Google's REST docs and has not yet been run against the live API |

- **`dimensions` must match the model.** The code checks every response and says so if it does not. The vector index has the
  same dimension (Neo4j allows 1 to 4096; above 2048 needs Neo4j 5.18 or later).
- **One model per set of vectors.** Vectors from different models cannot be compared. To change model or dimensions: call
  `vector.drop_vector_indexes()`, then `vector.create_vector_indexes(<dimensions>)`, then run `embed_nodes` again for every database.
- **Two models in one graph.** Give each its own `name` in the config (for example `name: voyage`). Indexes and vectors are stored
  per name (`table_embedding_voyage`, property `embedding_voyage`), the default name keeps the plain ones, and
  `embed_nodes` and `retrieve_seed_tables` use the name of the embedder you pass. Switch with `SCHEMAGRAPH_CONFIG`, or
  construct both with `vector.make_embedder(cfg)`. This lets you compare models on the same questions.
- Retrieval Cypher (`db.index.vector.queryNodes`) needs Neo4j 5.13 or later and has not yet been run on a real server:
  `SG_TEST_NEO4J=1 pytest tests/test_retrieve_neo4j.py`.

## Running the evaluation

```bash
pip install -e '.[eval]'     # pandas, only needed for run_eval.summarize()
python -m schemagraph.eval.run_eval --questions schemagraph/eval/questions/adventureworks.jsonl --check-gold
python -m schemagraph.eval.run_eval --questions schemagraph/eval/questions/adventureworks.jsonl \
    --modes full retrieval graph --out results/RUN.jsonl --run-id RUN --llm-model <model>
python -m schemagraph.eval.run_eval --summarize results/RUN.jsonl
```

### Compare retrieval settings end to end

Retrieval recall says whether the right tables arrive; only a model run says whether more tables help or hurt the SQL.
`--sweep` runs the same questions once per variant, `K` (top_k), `K+N` (also add up to N tables the seeds reference) or
`K+N:H` (looking H links away), and prints one row per variant with accuracy, schema recall, tables in the context,
tries, tokens and latency:

```bash
python -m schemagraph.eval.run_eval --questions schemagraph/eval/questions/adventureworks.jsonl \
    --modes graph --sweep 5 8 12 5+4:2 8+4:2 --out results/sweep.jsonl --run-id sweep --llm-model <model>
python -m schemagraph.eval.run_eval --compare-runs results/sweep.jsonl      # the table again, any time
```

Each variant is its own run id (`sweep-k8+4h2`), so an interrupted sweep resumes and finished rows are never asked
twice. It prints how many model calls it may make first (questions x modes x variants). `--top-k` and `--expand N[:H]`
override the config for a single ordinary run. Use the test split for the final choice and the dev split while exploring. Dev questions are also the few-shot examples, so `run_eval` tells the pipeline to leave a question's own
example out of its prompt (`exclude_example_ids`); without that, a dev run would show the model the answer.

### Evaluate on BIRD

BIRD's eleven databases are small (3 to 13 tables each) and each question reads 2.3 tables on average, so a `top_k` of
8 hands the model the whole schema for 49 of the 60 questions and tells you nothing about retrieval. Use `top_k` 2 to 5
here. Build only the eleven BIRD graphs (not `--all`, which would also pick up `bird_staging`; drop that database once
the split is done):

```bash
NAMES=$(grep -v '^\(#\|$\)' scripts/bird_tables.tsv | cut -f1 | sort -u)       # the eleven db_ids
python scripts/build_graph.py $(sed 's/^/--db /' <<< "$NAMES") --skip-extraction   # per-db metadata files are in data/metadata/
```

1. **Held-out inference.** Hide every declared foreign key and see how many come back (needs the databases, for the
   value-overlap check). One run per database; `debit_card_specializing` declares none, so it only shows what inference adds:
   `for d in $NAMES; do python -m schemagraph.eval.infer_recovery --db $d --fraction 1.0 --seeds 1; done`
2. **Retrieval alone** (no LLM), all eleven databases in one report with a per-database table:
   `python scripts/retrieval_eval.py --questions-file schemagraph/eval/questions/bird_subset.jsonl --top-k 2 3 5 --label bird --out runs/bird.json`.
   A database with at most 20 tables must reach 100% at k=20; if one does not, the shared vector index is cutting
   candidates before the `db_id` filter and `DEFAULT_CANDIDATES` in `graph/retrieve.py` needs raising.
3. **End to end**, modes and settings:
   `python -m schemagraph.eval.run_eval --questions schemagraph/eval/questions/bird_subset.jsonl --modes full retrieval graph --top-k 3 --out results/bird-modes.jsonl --run-id bird-modes --llm-model <model>`
   and `--modes graph --sweep 2 3 5 2+2:1 3+2:1 --out results/bird-sweep.jsonl --run-id bird-sweep`, then `--compare-runs`.
   Run it once more with `--no-evidence` to see how much BIRD's hint is doing.

### Building the question files

`bird_subset.jsonl` is generated from BIRD's own questions (the `mini_dev_postgresql.json` that `scripts/load_bird.sh` copies
to `data/bird/`), not written by hand:

```bash
python -m schemagraph.eval.bird_questions --input data/bird/mini_dev_postgresql.json \
    --out schemagraph/eval/questions/bird_subset.jsonl --verify      # --verify needs the BIRD databases loaded
# check the question files (list them; fewshot_examples.jsonl is not a question file) and the few-shot examples:
python -m schemagraph.eval.question_set schemagraph/eval/questions/adventureworks.jsonl \
    schemagraph/eval/questions/bird_subset.jsonl --fewshot schemagraph/eval/questions/fewshot_examples.jsonl
# add --verify to also run every gold query against the loaded databases
```

With `--verify`, failed gold queries are printed grouped by message. If at least half of all questions fail for one
reason (typically "database ... does not exist"), the run exits 2 and writes nothing: fix the setup, not the questions.

It keeps 50 `test` and 10 `dev` questions by default (`--n-test`, `--n-dev`), spread over the databases and mixed 20/40/40
across tiers 1/2/3 (`--tier-mix`), chosen reproducibly from `--seed`. BIRD databases have a single schema, so the spec's
tier 3 (cross-schema) cannot occur; here a question's tier is the number of distinct tables in its gold query (1, 2, or 3
and more). `order_matters` is set when the gold query has a top-level `ORDER BY` and no `LIMIT 1` (stricter than BIRD's own
scorer, which ignores row order); those questions say so in `notes`. `--verify` runs every gold query first and drops those
that fail, time out, or return no rows. `question_set.py` also catches duplicate ids and questions, gold SQL that is not
a single safe SELECT, and any overlap with `fewshot_examples.jsonl` (`--fewshot`). The questions not chosen are a pool for
few-shot examples.

`adventureworks.jsonl` is hand-written: 50 `test` questions (10 / 15 / 25 at tiers 1 / 2 / 3) and 10 `dev` questions, each with
gold SQL that was run against a real AdventureWorks and checked to run, return rows and have one clear answer (ranking
questions have no ties at the cut-off). The tier is again the number of distinct tables in the gold query, the same rule as
for BIRD, so tier 3 here means three or more tables, not "crosses schemas". Date-dependent questions fit this sample's dates
(orders 2022-05 to 2025-06, hire dates 2006 to 2013); the `notes` field says where a question depends on a reading such as
"currently". They are drafts: read them and edit any whose wording or gold SQL you would not defend.

`fewshot_examples.jsonl` holds every `dev` question of both files, the only questions the model may see as examples
(`retrieve_examples` filters by database and refuses any example that is not `dev`). The checker flags a few-shot
example that is not a `dev` question, and any `test` question that also appears there.

`--check-gold` runs every gold query first and reports failures and queries that return no rows; a real run does the
same check before any model call. A run is resumable: repeat the command with the same `--run-id` and finished rows are
skipped. Ablations: `--no-inferred`, `--unweighted` (passed to `answer_question` as `planner_options`).

How answers are judged (`eval/compare.py`): results are compared as sets of rows, ignoring column names and column
order, ignoring row order unless the question sets `order_matters`, and comparing numbers by value. Different rounding is
a different answer, and a truncated result never matches, so gold queries must return fewer than 100,000 rows.

## Testing

```bash
pytest -q                                   # no databases needed
```

Tests that need a database are skipped unless you opt in:

| Switch | Enables |
| --- | --- |
| `SG_TEST_PG_DSN="host=127.0.0.1 port=5432 user=app dbname=postgres"` | Integration tests on throwaway Postgres databases. The user must be able to create databases. |
| `SG_TEST_ADVENTUREWORKS=1` | Checks against the real AdventureWorks database (also needs the config and passwords above). |
| `SG_TEST_NEO4J=1` | Graph tests against Neo4j. They use a throwaway `sgtest_<random>` db_id and delete it afterwards. |

Before you push, run once with everything you have available. A skipped test has not passed.

## Conventions

- **UIDs** are `{db_id}:{schema}.{table}[.{column}]`. Build them with the helpers in `models.py`; never parse them.
  Use `ColumnMeta.table_uid` to go from a column to its table.
- **`db_id`** is lowercase letters, digits and underscores, and equals the Postgres database name.
- **Dump models with `by_alias=True`.**
- **Secrets** come from environment variables. `config.yml` and `.env` are never committed.
- **Pure code first.** Decide-what-to-do logic is plain functions with tests that need no database;
  the database-touching functions stay thin. New modules follow that pattern.
- **Tests that break things on purpose:** when you add a rule, break it deliberately and confirm a test fails.
  A test that passes either way proves nothing.
- **Changing shared files** (`models.py`, `config.py`): tell the others first.

## Known limits

- Relationship inference was tuned on AdventureWorks. On BIRD the name rules can reach only 43 of the 87 declared single-column
  foreign keys (49%), against 86 of 89 (97%) on AdventureWorks: none in `codebase_community`, `student_club` or
  `thrombosis_prediction`, which name keys `userid`, `link_to_member` or a bare `id`. That is the ceiling before the data check;
  the recovered/wrong/extra counts need `infer_recovery` run with the databases (step 1 of "Evaluate on BIRD"). Rules changed to
  fix BIRD are no longer held out.
- Inference proposes single-column relationships only. Declared composite keys are extracted correctly.
- Two tables with identical key sets cannot be told apart by the data, so inference picks one deterministically.
- The Postgres wrapper is Postgres-only by choice; abstracting it for other engines was deferred.
