"""FastAPI routes (SPEC 3.5, 4 and 5.17). Thin: validate input, call the pipeline or the graph, map errors to HTTP codes.

    pip install -e '.[api]'
    uvicorn schemagraph.api.main:app --reload          # docs at /docs, schema at /openapi.json

React --> POST /api/ask --> answer_question --> PostgreSQL + Neo4j + the LLM. There is one code path: the same
`answer_question` the evaluation harness calls, so what the evaluation measures is what users get.

Status codes (SPEC 3.5): 200 for every completed pipeline run, including failures reported in the response's
`status` (so the page can still show the explanation and attempts); 400 for bad input (empty question, unknown
db_id, invalid mode or options); 404 for a table that is not in the graph; 503 when PostgreSQL, Neo4j, the embedding
service or the LLM cannot be reached. Error bodies are FastAPI's default {"detail": "..."}. The detail never contains
a URL, a key or a trace; those go to the server log.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager

import neo4j.exceptions
import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemagraph import db
from schemagraph.config import DB_ID_RE, get_config
from schemagraph.graph import driver, queries
from schemagraph.graph.vector import EmbeddingError, RateLimited
from schemagraph.models import (
    AskRequest,
    AskResponse,
    Contract,
    JoinPlan,
    SeedTable,
    TableSummary,
    TableWithRelationships,
)
from schemagraph.pipeline import llm
from schemagraph.pipeline.context import build_context, expand_options
from schemagraph.pipeline.pipeline import answer_question

log = logging.getLogger(__name__)

# The React dev server (SPEC 5.17). Change this list, not the middleware call, for another front-end address.
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


class NotConfigured(Exception):
    """A dependency that is optional to configure is not configured (reported by /api/health)."""


class DatabaseInfo(Contract):
    db_id: str
    table_count: int
    schema_count: int


class Health(Contract):
    postgres: str
    neo4j: str
    llm: str


class PlanResponse(Contract):
    seed_tables: list[SeedTable]
    join_plan: JoinPlan | None = None  # null outside graph mode


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    driver.close_driver()


app = FastAPI(title="SchemaGraph", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def bad_input(_: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI answers 422 by default; the SPEC says 400 for an invalid mode, a missing question and the like."""
    problems = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()[:3]]
    return JSONResponse(status_code=400, content={"detail": "; ".join(problems) or "Invalid request."})


def unavailable(what: str, exc: Exception, retry_after: float | None = None) -> HTTPException:
    """A 503 whose detail names the failing dependency but nothing about it; the cause goes to the log."""
    log.error("%s: %s: %s", what, type(exc).__name__, exc, exc_info=exc)
    headers = {"Retry-After": str(max(1, round(retry_after)))} if retry_after else None
    return HTTPException(status_code=503, detail=what, headers=headers)


@contextmanager
def mapped_errors() -> Iterator[None]:
    """Turn the failures the SPEC names into HTTP errors. Anything else is a bug and stays a 500."""
    try:
        yield
    except HTTPException:
        raise
    except queries.TableNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:  # the pipeline's own input checks (top_k, max_attempts, row_limit)
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except RateLimited as exc:
        raise unavailable("The embedding service is rate-limited; try again shortly.", exc,
                          getattr(exc, "retry_after", None)) from None
    except EmbeddingError as exc:
        raise unavailable("The embedding service failed or is unreachable.", exc) from None
    except llm.LLMError as exc:
        raise unavailable("The language model service failed or is unreachable.", exc) from None
    except psycopg.Error as exc:
        raise unavailable("PostgreSQL is unreachable or refused the request.", exc) from None
    except (neo4j.exceptions.DriverError, neo4j.exceptions.Neo4jError) as exc:
        raise unavailable("Neo4j is unreachable or refused the request.", exc) from None


def known_database(db_id: str) -> str:
    """400 unless db_id is one the server can query. Checked before anything is looked up or run."""
    if not DB_ID_RE.fullmatch(db_id) or db_id not in db.list_databases():
        raise HTTPException(status_code=400, detail=f"Unknown db_id {db_id!r}.")
    return db_id


def checked(request: AskRequest) -> AskRequest:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    known_database(request.db_id)
    return request


# ---------------------------------------------------------------------------
# Endpoints (SPEC 4). Plain `def` so FastAPI runs the blocking work in its thread pool.
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=Health)
def health(deep: bool = False) -> Health:
    """Is each dependency usable? Values: "ok", "error", or "not_configured" (LLM only).

    The LLM check only looks at the configuration unless `?deep=true`, which sends a one-line request;
    a health check polled every few seconds should not spend tokens.
    """

    def check(fn) -> str:
        try:
            fn()
            return "ok"
        except NotConfigured:
            return "not_configured"
        except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
            log.warning("health check failed: %s: %s", type(exc).__name__, exc)
            return "error"

    def llm_check() -> None:
        cfg = get_config().llm
        if cfg is None:
            raise NotConfigured
        try:
            cfg.api_key  # noqa: B018 - raises RuntimeError when the key's environment variable is unset
        except RuntimeError as exc:
            raise NotConfigured from exc
        if deep:
            llm.complete(cfg, [{"role": "user", "content": "Reply with the single word OK."}])

    return Health(postgres=check(db.list_databases), neo4j=check(driver.check_connection), llm=check(llm_check))


@app.get("/api/databases", response_model=list[DatabaseInfo])
def databases() -> list[DatabaseInfo]:
    """Databases that can be queried: in PostgreSQL *and* with a graph built (every mode reads the graph)."""
    found = []
    with mapped_errors():
        for db_id in db.list_databases():
            tables = queries.list_tables(db_id)
            if tables:
                schemas = {t.display_name.split(".", 1)[0] for t in tables}
                found.append(DatabaseInfo(db_id=db_id, table_count=len(tables), schema_count=len(schemas)))
    return found


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Question to SQL, result and explanation. Pipeline failures come back as 200 with `status` set."""
    with mapped_errors():
        return answer_question(checked(request))


@app.post("/api/plan", response_model=PlanResponse)
def plan(request: AskRequest) -> PlanResponse:
    """Retrieval and join planning only, no LLM call: what the model would be given, for debugging the graph."""
    with mapped_errors():
        checked(request)
        pipeline_config = get_config().pipeline
        options = request.options
        top_k = options.top_k if options and options.top_k is not None else pipeline_config.top_k
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        context, seeds = build_context(
            request.db_id, request.question, request.mode, top_k, expand=expand_options(pipeline_config)
        )
        return PlanResponse(seed_tables=seeds, join_plan=context.join_plan)


@app.get("/api/schema/{db_id}/tables", response_model=list[TableSummary])
def schema_tables(db_id: str) -> list[TableSummary]:
    with mapped_errors():
        return queries.list_tables(known_database(db_id))


@app.get("/api/schema/{db_id}/tables/{table_uid}", response_model=TableWithRelationships)
def schema_table(db_id: str, table_uid: str) -> TableWithRelationships:
    with mapped_errors():
        return queries.get_table_detail(known_database(db_id), table_uid)
