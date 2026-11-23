# SchemaGraph

Text-to-SQL for PostgreSQL. You ask a question in plain English; SchemaGraph finds the tables it is about, works out how
to join them, asks a language model for one read-only `SELECT`, checks it, runs it, and returns the SQL, the rows and an
explanation of which tables and joins were used.

Its distinguishing idea is a **schema graph**. Instead of sending the model the whole schema, it stores tables, columns
and foreign keys in Neo4j, finds the relevant tables by vector search, and plans the cheapest join path between them, so
the model sees a small, connected slice of the schema even in databases with dozens of tables across several schemas.

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Loading data and building the graph](#loading-data-and-building-the-graph)
- [Using the API](#using-the-api)
- [Web app](#web-app)
- [Running everything in Docker](#running-everything-in-docker)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Development notes](#development-notes)
- [Known limits](#known-limits)

## How it works

```
                 build time                                          per question
Postgres ──extract──▶ metadata JSON ──ingest──▶ Neo4j graph ──▶ retrieve ──▶ plan joins ──▶ LLM ──▶ check ──▶ run read-only
(catalog)  + infer     data/metadata/          tables, columns,  (vector     (Steiner      (SQL)   (one       (sg_readonly
           missing                             foreign keys,      search)    tree)                 SELECT)     role)
           foreign keys                        embeddings
```

**Build time** (once per database, from the command line):

1. **Extract** tables, columns, primary keys, foreign keys and unique constraints from the Postgres catalog.
2. **Infer** foreign keys the database does not declare, from matching names and types confirmed by overlapping values.
   Each inferred relationship gets a confidence (at most 0.95) and the evidence for it.
3. **Ingest** the result into Neo4j: `(:Schema)-[:CONTAINS]->(:Table)-[:HAS_COLUMN]->(:Column)`, with
   `(:Column)-[:REFERENCES]->(:Column)` from each foreign-key column to the column it references.
4. **Weight** every relationship: a declared join costs 1.0 walked from the referencing table and 2.0 walked the other way
   (where it can multiply rows); inferred joins cost more the less certain they are.
5. **Embed** every table and column with the configured embedding model, for vector search.

**Per question** (`answer_question` in `schemagraph/pipeline/pipeline.py`, used by both the API and the evaluation):

1. **Retrieve** the `top_k` tables closest to the question, matching on table or column embeddings.
2. **Plan joins** (graph mode): find the cheapest tree of joins connecting those tables, adding any bridge tables in
   between. The planner uses an exact Steiner-tree solver for up to 7 tables and an approximation beyond.
3. **Generate** SQL with an OpenAI-compatible chat model, given the selected table definitions, the planned joins and a
   few similar example questions. Inferred relationships are marked as guesses with their confidence.
4. **Check** that the SQL is exactly one read-only `SELECT` (no writes, no `SELECT INTO`, no locking, no dangerous
   functions), ask PostgreSQL to plan it, and on failure feed the error back to the model, up to `max_attempts` times.
5. **Run** it as the `sg_readonly` login, which can only read, inside a read-only transaction with a statement timeout,
   returning at most `row_limit` rows.
6. **Explain** which tables were used and why, the join path, any warnings, and a one-sentence summary.

There are three **modes**, so the graph's contribution can be measured:

| Mode | What the model is given |
| --- | --- |
| `full` | The whole schema |
| `retrieval` | The `top_k` retrieved tables, no join planning |
| `graph` (default) | The retrieved tables plus the planned joins and bridge tables |

## Repository layout

| Path | What it holds |
| --- | --- |
| `schemagraph/models.py` | Every shared data contract (Pydantic), plus the UID helpers |
| `schemagraph/config.py` | Loading `config.yml` and `.env`; Postgres, Neo4j, embedding, LLM and pipeline settings |
| `schemagraph/db/` | PostgreSQL access: `list_databases`, `run_readonly_query`, `explain`, `run_admin_query` |
| `schemagraph/sql_validation.py` | The check that accepts one read-only `SELECT` and rejects everything else |
| `schemagraph/extraction/` | Catalog queries (`sql/`), metadata extraction, relationship inference, `build_metadata` |
| `schemagraph/graph/` | Neo4j driver, ingest, edge weights, queries, embeddings (`enrich`, `vector`), retrieval, join planner |
| `schemagraph/pipeline/` | Prompt context, few-shot examples, LLM client, SQL generation, validation, explanation, `answer_question` |
| `schemagraph/api/main.py` | The FastAPI app |
| `schemagraph/eval/` | Evaluation: question sets, result comparison, schema metrics, `run_eval`, inference recovery |
| `frontend/` | The React + TypeScript web app (Vite) |
| `scripts/` | Data loaders, Postgres role setup, `build_graph`, smoke test, retrieval evaluation |
| `data/metadata/` | Extracted metadata JSON, one file per database |
| `contracts/examples/` | Sample metadata, join plan and API response, checked against the models by the tests |
| `tests/` | Python tests (`pytest`) |
| `docker-compose.yml`, `Dockerfile`, `docker/` | Postgres, Neo4j, the API and the web app as containers |
| `project overview/SPEC.md` | The original design; section numbers quoted in the code refer to it |

## Quick start

Requirements: Python 3.11+, Docker, `psql`/`pg_dump` 15 or newer, Node 20.19+ (for the web app), an API key for an
embedding model and a chat model (or local OpenAI-compatible servers for both).

```bash
# 1. Python package
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,eval]"

# 2. Secrets: copy and fill in the passwords and the provider key(s) config.yml uses
cp .env.example .env

# 3. Postgres 17 and Neo4j 5, bound to 127.0.0.1
docker compose up -d db neo4j

# 4. Load a database (AdventureWorks; see "Loading data" for BIRD)
git clone https://github.com/lorint/AdventureWorks-for-Postgres adventureworks-for-postgres
set -a; source .env; set +a                                  # the shell scripts need the passwords exported
PGHOST=127.0.0.1 PGPORT=5432 PGUSER=app PGPASSWORD=$DB_PASSWORD \
    scripts/load_adventureworks.sh --repo ./adventureworks-for-postgres

# 5. Build its graph: extract, infer, ingest, weights, vector indexes, embeddings, verify
python scripts/build_graph.py --db adventureworks --rebuild

# 6. Check it end to end, then start the API and the web app
python scripts/smoke_test.py --db adventureworks
uvicorn schemagraph.api.main:app --reload                    # http://localhost:8000/docs
cd frontend && npm install && npm run dev                    # http://localhost:5173
```

Ask a question from the command line:

```bash
curl -s localhost:8000/api/ask -H 'Content-Type: application/json' \
    -d '{"question": "How many employees are there in each department?", "db_id": "adventureworks"}'
```

## Configuration

Two files configure everything:

- **`config.yml`** holds settings and the *names* of environment variables that hold secrets, never the secrets
  themselves. It is committed. Point `SCHEMAGRAPH_CONFIG` at another file to switch configurations; a relative path is
  tried from the current folder, then from the repository root.
- **`.env`** (git-ignored) holds the passwords and API keys. The Python code loads it from the repository root
  automatically; a variable already set in the shell wins. Docker Compose reads it too. Shell scripts do not, so export
  it first with `set -a; source .env; set +a`. `SCHEMAGRAPH_ENV_FILE` points at a different file (empty turns loading off).

`.env` needs:

| Variable | Used by |
| --- | --- |
| `DB_PASSWORD`, `APP_DB_PASSWORD` | The Postgres container's superuser `app`, read only when its data volume is first created |
| `PG_ADMIN_PASSWORD` | `sg_admin`: catalog extraction, inference, data loading. Never runs model SQL |
| `PG_READONLY_PASSWORD` | `sg_readonly`: SELECT only, read-only transactions, statement timeout. Runs every generated query |
| `NEO4J_PASSWORD` | Neo4j (at least 8 characters; read once, when its volume is first created) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `VOYAGE_API_KEY` | Whichever providers `config.yml` names. Leave unused ones out rather than blank: an empty value counts as set |

The two Postgres logins are the main safety measure: even if a generated query got past the SQL check, the database
itself would refuse to change anything. `scripts/create_roles.sql` creates them.

`config.yml` sections (commented examples for each provider are in the file):

| Section | Settings |
| --- | --- |
| `postgres` | `host`, `port`, `sslmode`, `statement_timeout_ms`, and the `admin` / `readonly` logins (`user`, `password_env`) |
| `neo4j` | `uri`, `user`, `password_env`, `database` |
| `embedding` | `provider` (`openai`, `voyage`/`anthropic`, `gemini`), `model`, `dimensions`, `send_dimensions`, `name`, `api_key_env`, `query_cache`, rate-limit options (`max_wait_s`, `requests_per_minute`, `batch_size`), `include_relationships` |
| `llm` | Any OpenAI-compatible `/chat/completions` endpoint: `base_url`, `model`, `api_key_env`, `temperature`, `max_tokens`, `timeout_s`, `max_retries`, `max_wait_s` |
| `pipeline` | Defaults for each question, below |

`pipeline` settings (a request's `options` can override the first three):

| Setting | Default | Meaning |
| --- | --- | --- |
| `top_k` | 5 | Tables retrieval returns |
| `max_attempts` | 3 | SQL generation tries, each fed the previous error |
| `row_limit` | 100 | Rows returned |
| `examples_k` | 3 | Few-shot examples in the prompt (development questions only) |
| `fewshot_path` | `schemagraph/eval/questions/fewshot_examples.jsonl` | Where the examples come from |
| `expand_seeds`, `expand_hops` | 0, 1 | Graph mode: also add up to N tables the retrieved tables reference, H links away |
| `planner_selection` | `all` | Graph mode: `all` connects every retrieved table it can; `prize` connects only those worth their joins |
| `prize_scale` | 3.0 | With `prize`: how much join cost the best-matching table is worth (3.0 is about two joins) |

**Embedding models.** Vectors from different models cannot be compared, and `dimensions` must match both the model and
the Neo4j vector index (the code checks every response). To keep two models in one graph, give each config its own
`embedding.name`; vectors and indexes are stored per name. To change the model or dimensions under the same name, drop
the indexes (`vector.drop_vector_indexes(name=...)`) and rebuild. Vector search needs Neo4j 5.13 or later.

## Loading data and building the graph

Neither dataset is stored in the repository.

| Dataset | What it is | Source |
| --- | --- | --- |
| **AdventureWorks** | Microsoft's sample company database: 68 tables in 5 schemas, nearly every foreign key declared | The Postgres port at <https://github.com/lorint/AdventureWorks-for-Postgres>; `load_adventureworks.sh` downloads Microsoft's CSV data (about 17 MB) |
| **BIRD mini-dev** | A public Text-to-SQL benchmark: 500 questions over 11 small databases, with PostgreSQL gold queries | `minidev.zip`, linked from <https://github.com/bird-bench/mini_dev>; the files used are `MINIDEV_postgresql/BIRD_dev.sql` and `mini_dev_postgresql.json` |

Check each dataset's licence before redistributing any of it.

```bash
set -a; source .env; set +a
export PGHOST=127.0.0.1 PGPORT=5432 PGUSER=app PGPASSWORD=$DB_PASSWORD      # a superuser

scripts/load_adventureworks.sh --repo ./adventureworks-for-postgres         # needs git and Ruby
scripts/load_bird.sh --zip ~/Downloads/minidev.zip                          # one database per BIRD db_id
scripts/load_bird.sh --from-staging --force    # if every BIRD table is already in a `bird_staging` database
```

BIRD's dump puts all eleven databases in one schema; `load_bird.sh` splits them using `scripts/bird_tables.tsv` and
stops without changing anything if that file and the dump disagree. Both loaders create the databases, run
`scripts/create_roles.sql`, run `ANALYZE` and check that `sg_readonly` can read but not write. A database loaded any
other way needs `psql -d <db> -f scripts/create_roles.sql` as a superuser.

**Build the graph** with one command per database or all of them:

```bash
python scripts/build_graph.py --db adventureworks --rebuild            # everything, from scratch
python scripts/build_graph.py --db adventureworks --skip-extraction    # reuse data/metadata/adventureworks.json
python scripts/build_graph.py --all --skip-enrichment                  # no embedding model needed
```

Other flags: `--skip-inference`, `--reembed`, and inference tuning (`--sample-n`, `--min-overlap`, `--min-confidence`).
The steps also run on their own: `python -m schemagraph.extraction.build_metadata --db <db>` writes the metadata file,
and `python -m schemagraph.graph.ingest --db <db> --rebuild` loads it into Neo4j.

Before changing anything, `build_graph` checks that Neo4j answers and that the embedding model returns vectors of the
configured size (exit 2 if not). Exit 1 means a database failed or its graph does not match its metadata; exit 3 means
embedding stopped on a rate limit. Embeddings are saved as they are made, so rerunning the same command **without
`--rebuild`** (which clears them) embeds only what is missing. `--rebuild` clears a database's nodes, including its
embeddings.

**Free-tier keys.** Rate-limited requests wait as the service asks, up to `embedding.max_wait_s`, then stop cleanly.
`requests_per_minute` spaces requests out. With `embedding.query_cache` set, each question is embedded once and reused
by every mode, setting and rerun.

**Check a build** before any long run:

```bash
python scripts/smoke_test.py --db adventureworks                      # five built-in questions, all three modes
python scripts/smoke_test.py --db adventureworks --question "How many employees are there?" --mode graph
python scripts/smoke_test.py --db adventureworks --no-llm             # retrieval and join planning only
```

It prints what retrieval found, the join plan, the tables the model was given and, with an `llm` section, the SQL and
whether it ran. With `--questions-file` it also checks answers against the gold results.

## Using the API

```bash
uvicorn schemagraph.api.main:app --reload       # http://localhost:8000, OpenAPI docs at /docs
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | `ok` / `error` for `postgres` and `neo4j`; `ok`, `error` or `not_configured` for `llm`. Add `?deep=true` for a real model call |
| `GET` | `/api/databases` | Databases that exist in Postgres and have a graph, with table and schema counts |
| `POST` | `/api/ask` | Question to SQL, result and explanation |
| `POST` | `/api/plan` | Retrieval and join planning only, no model call (same body as `/api/ask`) |
| `GET` | `/api/schema/{db_id}/tables` | The database's tables |
| `GET` | `/api/schema/{db_id}/tables/{table_uid}` | One table with its columns and relationships |

**Request** (`/api/ask`):

```json
{
  "question": "What is the total order value for each customer's name?",
  "db_id": "adventureworks",
  "mode": "graph",
  "options": { "top_k": 5, "max_attempts": 3, "row_limit": 200 }
}
```

`mode` and `options` are optional. `top_k` may be 1–50, `max_attempts` 1–5, `row_limit` 1–100,000.

**Response** (abridged):

```json
{
  "status": "success",
  "sql": "SELECT p.firstname, p.lastname, SUM(soh.totaldue) ... GROUP BY p.firstname, p.lastname",
  "result": { "columns": ["firstname", "lastname", "total"], "rows": [["Jon", "Yang", "8248.99"]],
              "row_count": 1, "truncated": false },
  "explanation": {
    "summary": "Adds up the value of every order for each customer.",
    "tables": [{ "uid": "adventureworks:sales.customer", "display_name": "sales.customer",
                 "role": "bridge", "reason": "Needed to connect the retrieved tables." }],
    "join_path": [{ "from": "sales.salesorderheader.customerid", "to": "sales.customer.customerid",
                    "source": "declared", "text": "Declared foreign-key relationship." }],
    "interpretation": null,
    "warnings": []
  },
  "join_plan": { "status": "full", "algorithm": "exact_steiner", "tables": ["..."], "joins": ["..."] },
  "attempts": [{ "sql": "SELECT ...", "error": null }],
  "metrics": { "input_tokens": 1834, "output_tokens": 212, "attempt_count": 1,
               "latency_ms": { "retrieval": 120, "planning": 45, "generation": 2310, "total": 3925 } }
}
```

- **`status`**: `success`, `invalid_sql` (failed validation on every attempt), `execution_error`, `no_schema` (nothing
  relevant retrieved) or `unsafe_sql` (not a single read-only SELECT).
- **`result.rows`**: every value is a string or `null`; `truncated` is true if more than `row_limit` rows existed.
- **`explanation.tables[].role`**: `seed` (matched the question), `bridge` (added to connect them) or `context`.
- **`join_plan`** (graph mode): `status` is `full`, `partial` (some tables left out), `single_table` or `none` (the
  prompt falls back to the retrieved tables). It lists `seed_tables`, `bridge_tables`, `tables`, `joins` (each with
  `source`, `confidence`, `cardinality`, `traversal`, `weight`), `total_cost`, `unreachable_seeds` (no relationship
  reaches them; still shown to the model with a note), `dropped_seeds` (left out for cost) and `alternatives`.
- **`attempts`**: every generated query in order, with the database error for those that failed.

**HTTP codes**: `200` for every completed run, including failures reported in `status`; `400` for bad input (empty
question, unknown `db_id`, invalid `mode` or `options`); `404` for an unknown table; `503` when PostgreSQL, Neo4j, the
embedding service or the model cannot be reached, naming which (with `Retry-After` when rate limited). Errors are
`{"detail": "..."}`.

## Web app

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173; expects the API on http://localhost:8000
VITE_USE_MOCKS=true npm run dev      # no backend: sample answers from src/mocks
npm test                             # unit and component tests
npm run build                        # type-check, then build to dist/
npm run gen:types                    # regenerate src/api/types.ts from the running API's /openapi.json
```

Pick a database, ask a question, and the page shows the result table, the SQL, the tables used and why, and the join
path: filled stops are tables that matched the question, rings are tables passed through, solid lines are declared joins
and dashed amber lines are inferred ones. The developer view (header toggle, or `?dev=1`) adds the mode selector, a
"Plan only" button (`/api/plan`, no model call), request metrics and the full join plan. In mock mode, a question
starting with `mock:invalid_sql`, `mock:no_schema`, `mock:bad` or `mock:unavailable` shows that failure.

If the API runs elsewhere, set `VITE_API_URL` (see `frontend/.env.example`) and add the web app's address to
`CORS_ORIGINS` in `schemagraph/api/main.py`, which allows only `localhost:5173` and `127.0.0.1:5173`.

## Running everything in Docker

| Service | Address | What |
| --- | --- | --- |
| `frontend` | <http://localhost:5173> | The web app, served by nginx |
| `api` | <http://localhost:8000> (docs at `/docs`) | The FastAPI app |
| `db` | `127.0.0.1:5432` | Postgres 17 |
| `neo4j` | <http://localhost:7474>, bolt `127.0.0.1:7687` | Neo4j 5 |

```bash
docker compose up -d db neo4j      # databases first; load data and build graphs from the host as above
docker compose up -d --build       # then the API and the web app
curl localhost:8000/api/health     # postgres and neo4j should be "ok"
```

- Compose reads `.env` and refuses to start the API without `PG_ADMIN_PASSWORD`, `PG_READONLY_PASSWORD` and
  `NEO4J_PASSWORD`. Provider keys are passed through only when set.
- The API uses your `config.yml`, mounted into the container: after editing it, `docker compose restart api` is enough.
  Its entrypoint rewrites `127.0.0.1`/`localhost` database addresses to the `db` and `neo4j` services in a copy, so one
  config serves both the host and the containers.
- The web app's API address is compiled in and is the address your browser uses, so it defaults to
  `http://localhost:8000`. Change it with `VITE_API_URL=... docker compose build frontend`.
- An empty database picker means no graph has been built for any database.
- All ports are bound to `127.0.0.1`. The query-embedding cache lives in the `querycache` volume.

## Evaluation

The evaluation runs real questions with known answers through `answer_question`, the same path the API uses, and
records for each question and mode whether the result matched, which tables reached the prompt, tokens and latency.

```bash
Q=schemagraph/eval/questions/adventureworks.jsonl
python -m schemagraph.eval.run_eval --questions $Q --check-gold                 # run every gold query first
python -m schemagraph.eval.run_eval --questions $Q --modes full retrieval graph \
    --out results/run.jsonl --run-id run1 --llm-model <model>
python -m schemagraph.eval.run_eval --summarize results/run.jsonl              # by mode and tier
```

Each result row records `correct`, `status`, the SQL, `gold_tables` and `context_tables`, `schema_recall` (the share of
the tables the gold query needs that reached the prompt), `schema_precision` (the share of the prompt's tables the gold
query uses), bridge tables needed and recovered, the plan status, tokens, latency and attempts. A run is resumable: rerun
with the same `--run-id` and finished rows are skipped. If the embedding quota runs out, it stops cleanly (exit 3).

**Answers are judged** as sets of rows, ignoring column names and order, ignoring row order unless the question sets
`order_matters`, and comparing numbers by value. A truncated result never matches.

**Comparing settings.** `--sweep` runs the same questions once per retrieval setting, `K` (`top_k`), `K+N` (also add N
referenced tables) or `K+N:H` (looking H links away), each under its own run id, then prints one row per setting:

```bash
python -m schemagraph.eval.run_eval --questions $Q --modes graph --sweep 5 8 5+4:2 \
    --out results/sweep.jsonl --run-id sweep --llm-model <model>
python -m schemagraph.eval.run_eval --compare-runs results/sweep.jsonl
```

Other options for a single run: `--top-k`, `--expand N[:H]`, `--selection all|prize`, `--prize-scale N`, the planner
ablations `--no-inferred` (ignore inferred relationships) and `--unweighted` (every join costs 1), `--split test|dev|all`,
`--no-evidence` (leave out BIRD's hints) and `--skip-bad-gold`. To compare seed selection, run the same questions under
different run ids (for example `--selection prize --prize-scale 2`, `3`, `4`) into one file and `--compare-runs` it:
the goal is the same recall and accuracy with higher precision and fewer tables.

**Retrieval on its own**, with no model and no Postgres:

```bash
python scripts/retrieval_eval.py --db adventureworks --questions-file $Q --label run-a --out runs/a.json
python scripts/retrieval_eval.py --compare runs/a.json runs/b.json
```

It reports recall@k, how well scores separate relevant tables from the rest, and what the planner would give the model
for each seed strategy. Use it to choose `top_k` and to compare embedding models. `--expand` picks the seed expansions
to try (`0` for none, `N` or `N:H`), and `--prize-scale S [S ...]` also plans every strategy with prize-based selection
at each scale, so scales can be compared in seconds before spending model calls on the best ones:

```bash
python scripts/retrieval_eval.py --db adventureworks --questions-file $Q --top-k 5 --expand 0 --prize-scale 2 3 4 6 8
```

**Relationship inference**: `python -m schemagraph.eval.infer_recovery --db <db> --fraction 1.0 --seeds 1` hides the
declared foreign keys and measures how many inference recovers.

**Question files** are JSONL, one question per line:

```json
{"id": "aw-017", "db_id": "adventureworks", "question": "...", "gold_sql": "SELECT ...", "tier": 3, "split": "test",
 "order_matters": false, "evidence": null, "author": "...", "notes": ""}
```

- `schemagraph/eval/questions/adventureworks.jsonl`: 50 `test` and 10 `dev` questions, hand-written and checked against
  a real AdventureWorks. `tier` is the number of distinct tables the gold query reads (1, 2, or 3 and more).
- `schemagraph/eval/questions/bird_subset.jsonl`: 50 `test` and 10 `dev` questions sampled reproducibly from BIRD
  mini-dev by `python -m schemagraph.eval.bird_questions --input data/bird/mini_dev_postgresql.json --out ... --verify`.
  BIRD databases are small (3 to 13 tables), so use `top_k` 2 to 5 there.
- `schemagraph/eval/questions/fewshot_examples.jsonl`: the only questions the model may see as examples. They must all be
  `dev` questions; `run_eval` keeps a question's own example out of its prompt.
- Check the files with `python -m schemagraph.eval.question_set <question files> --fewshot <fewshot file>` (add
  `--verify` to run every gold query).

## Testing

```bash
pytest -q                                   # no databases needed
```

Tests that need a live service are skipped unless enabled:

| Switch | Enables |
| --- | --- |
| `SG_TEST_PG_DSN="host=127.0.0.1 port=5432 user=app dbname=postgres"` | Integration tests on throwaway Postgres databases (the user must be able to create databases) |
| `SG_TEST_ADVENTUREWORKS=1` | Checks against the real AdventureWorks database |
| `SG_TEST_NEO4J=1` | Graph tests against Neo4j, on a throwaway `db_id` deleted afterwards |

The tests never read your `.env`. A skipped test has not passed, so run with every switch you can before relying on a
change. The web app's tests run with `npm test` in `frontend/`.

## Development notes

- **UIDs** are `{db_id}:{schema}.{table}[.{column}]`. Build and display them with the helpers in `models.py`
  (`display_name`); never split one yourself, because names can contain dots.
- **`db_id`** is lowercase letters, digits and underscores, and equals the Postgres database name.
- **Dump models with `by_alias=True`**: `schema` and `from` are aliased because they clash with Python names.
- **One entry point.** The API and the evaluation both call `answer_question`, so the evaluation measures what users get.
- **One way to run model SQL.** It must pass `check_select` and then run through `run_readonly_query`. Nothing else
  executes generated SQL, and the admin login never does.
- **Pure logic first.** Decisions are plain functions tested without a database; the code that touches a database
  stays thin. The join planner, for example, runs on in-memory edges (`plan_from_edges`).
- **`models.py` is the contract** between the backend, the evaluation and the web app (whose `types.ts` is generated
  from the API). `contracts/examples/` holds sample data the tests keep in step with it.

## Known limits

- Relationship inference was tuned on AdventureWorks, where its name rules can reach 86 of 89 declared single-column
  foreign keys (97%). On BIRD they reach 43 of 87 (49%), and none in `codebase_community`, `student_club` or
  `thrombosis_prediction`, which name keys `userid`, `link_to_member` or a bare `id`.
- Inference proposes single-column relationships only (declared composite keys are extracted correctly), and cannot
  tell apart two tables with identical key sets.
- Retrieval searches one vector index shared by every database and filters by database afterwards, so a small database
  can get fewer than `top_k` tables when many databases are loaded. `DEFAULT_CANDIDATES` in `graph/retrieve.py` sets how
  many candidates are fetched.
- With `planner_selection: prize`, at most 7 retrieved tables are considered; `prize_scale` has not been tuned yet.
- LLM-written descriptions of tables and columns are not generated yet.
- Database access is PostgreSQL only.
