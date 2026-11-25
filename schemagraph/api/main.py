"""A VERY SMALL FastAPI example for the group's Text-to-SQL project.

This file is the bridge between React and the team's Python pipeline.
It starts in DEMO_MODE so you can test it without PostgreSQL, Neo4j, or an LLM.
"""

# CHANGED #1: Import FastAPI, like your old inventory backend.
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# CHANGED #2: Use the request and response types your team ALREADY defined.
# No need to create new Pydantic schemas: they're in schemagraph/models.py.
from schemagraph.models import AskRequest, AskResponse

# Keep this True until the other backend services are ready.
# True  = send a fixed SAMPLE answer from the team's JSON file.
# False = call your teammate's real SQL-generation pipeline.
DEMO_MODE = True

# SAME AS YOUR OLD BACKEND: create the FastAPI application.
app = FastAPI(title="Simple Text-to-SQL API")

# CHANGED #3: Allow the React development server to talk to this API.
# React runs on port 5173; FastAPI runs on port 8000.
# Limit origins to local development instead of allowing every website.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# SAME IDEA AS YOUR OLD @app.get("/") health check.
# Visit http://localhost:8000/ to see if FastAPI started.
@app.get("/")
def home():
    return {"message": "Text-to-SQL backend is running", "demo_mode": DEMO_MODE}


# CHANGED #4: Your frontend uses POST /api/ask, NOT POST /items.
# FastAPI reads the JSON body and creates an AskRequest object automatically.
# response_model also checks that we return the agreed AskResponse format.
@app.post("/api/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # Users should not be able to send a blank or all-spaces question.
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")

    # CHANGED #5: SAMPLE MODE: use the example JSON the team has already made.
    # This lets your React app demonstrate a complete response today.
    if DEMO_MODE:
        # The saved example is only for this database and mode.
        if request.db_id != "adventureworks" or request.mode != "graph":
            raise HTTPException(
                status_code=400,
                detail="Demo mode supports only adventureworks with graph mode.",
            )

        # From schemagraph/api/main.py, go up to the repository root.
        # Then open contracts/examples/ask_response.sample.json.
        example_file = (
                Path(__file__).resolve().parents[2]
                / "contracts" / "examples" / "ask_response.sample.json"
        )
        with example_file.open(encoding="utf-8") as file:
            sample = json.load(file)

        # No LLM is connected! Do not pretend other questions were answered.
        if request.question.strip().casefold() != sample["question"].casefold():
            raise HTTPException(
                status_code=400,
                detail="Sample mode supports only: " + sample["question"],
            )

        # Return the saved SQL, table results and explanations as JSON.
        sample["request_id"] = str(uuid4())
        sample["question"] = request.question.strip()
        return sample

    # CHANGED #6: REAL MODE: this is the important handoff to your teammate.
    # Their function handles Neo4j, the LLM, SQL validation, and PostgreSQL.
    # We import it here so DEMO_MODE works without those dependencies running.
    from schemagraph.pipeline.pipeline import answer_question

    return answer_question(request)


import logging
import neo4j.exceptions
import psycopg
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from schemagraph import db
from schemagraph.config import DB_ID_RE, get_config
from schemagraph.graph import driver, queries
from schemagraph.graph.vector import EmbeddingError, RateLimited
from schemagraph.models import JoinPlan, SeedTable, TableSummary, TableWithRelationships
from schemagraph.pipeline import llm
from schemagraph.pipeline.context import build_context

# Real error messages go here
log = logging.getLogger(__name__)

# Limits on the options a user can send, so one request can't run up the LLM bill
MAX_TOP_K = 50
MAX_ATTEMPTS = 5          # each attempt is a paid LLM call
MAX_ROW_LIMIT = 100_000

class DatabaseInfo(BaseModel):
    db_id: str
    table_count: int
    schema_count: int
class Health(BaseModel):
    postgres: str   # "ok" or "error"
    neo4j: str      # "ok" or "error"
    llm: str        # "ok", "error" or "not_configured"
class PlanResponse(BaseModel):
    seed_tables: list[SeedTable]
    join_plan: JoinPlan | None = None   # None outside graph mode

# FastAPI runs one of these when any route raises that kind of error,
# so /api/ask gets them too. Any other error is a bug and stays a 500.

def service_down(message, error, retry_after=None):
    # 503 means a service we depend on is down.
    # The real error can contain URLs, passwords or keys, so it only goes to the log.
    log.error("%s (%s: %s)", message, type(error).__name__, error)
    headers = None
    if retry_after:
        headers = {"Retry-After": str(max(1, round(retry_after)))}
    return JSONResponse(status_code=503, content={"detail": message}, headers=headers)
@app.exception_handler(llm.LLMError)
def llm_down(request, error):
    return service_down("Error", error)
@app.exception_handler(RateLimited)
def embedding_rate_limited(request, error):
    return service_down("Error", error, error.retry_after)
@app.exception_handler(EmbeddingError)
def embedding_down(request, error):
    return service_down("Error", error)
@app.exception_handler(psycopg.Error)
def postgres_down(request, error):
    return service_down("Error", error)
@app.exception_handler(neo4j.exceptions.DriverError)
@app.exception_handler(neo4j.exceptions.Neo4jError)
def neo4j_down(request, error):
    return service_down("Error", error)
@app.exception_handler(queries.TableNotFound)
def table_not_found(request, error):
    return JSONResponse(status_code=404, content={"detail": str(error)})
@app.exception_handler(ValueError)
def bad_value(request, error):
    # The pipeline raises ValueError for bad options
    return JSONResponse(status_code=400, content={"detail": str(error)})
@app.exception_handler(RequestValidationError)
def bad_request_body(request, error):
    # FastAPI normally answers bad JSON with 422. The specifications asks for 400.
    first = error.errors()[0]
    where = ".".join(str(part) for part in first["loc"] if part != "body")
    return JSONResponse(status_code=400, content={"detail": f"{where}: {first['msg']}"})

# the new routes check their input before anything runs

def check_db_id(db_id):
    # Only allow databases the read-only login can actually use.
    if not DB_ID_RE.fullmatch(db_id) or db_id not in db.list_databases():
        raise HTTPException(status_code=400, detail=f"Unknown db_id {db_id!r}.")
def check_limit(name, value, biggest):
    # None means it uses the default from config.yml
    if value is not None and (value < 1 or value > biggest):
        raise HTTPException(status_code=400, detail=f"options.{name} must be between 1 and {biggest}.")
def check_question(request):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    if request.options is not None:
        check_limit("top_k", request.options.top_k, MAX_TOP_K)
        check_limit("max_attempts", request.options.max_attempts, MAX_ATTEMPTS)
        check_limit("row_limit", request.options.row_limit, MAX_ROW_LIMIT)


# get /api/health checks PostgreSQL, Neo4j and the LLM.
# Normal checks don't message the LLM, so calling this often costs nothing.
@app.get("/api/health", response_model=Health)
def health(deep: bool = False):
    status = {"postgres": "ok", "neo4j": "ok", "llm": "ok"}
    try:
        db.list_databases()
    except Exception as error:
        log.warning("health: PostgreSQL failed: %s", error)
        status["postgres"] = "error"
    try:
        driver.check_connection()
    except Exception as error:
        log.warning("health: Neo4j failed: %s", error)
        status["neo4j"] = "error"
    try:
        llm_config = get_config().llm
        if llm_config is None:
            status["llm"] = "not_configured"
        elif not has_api_key(llm_config):
            status["llm"] = "not_configured"   # the key's environment variable is not set
        elif deep:
            llm.complete(llm_config, [{"role": "user", "content": "Reply with the single word OK."}])
    except Exception as error:
        log.warning("health: LLM failed: %s", error)
        status["llm"] = "error"
    return status

def has_api_key(llm_config):
    try:
        llm_config.api_key   # raises RuntimeError if the environment variable is not set
        return True
    except RuntimeError:
        return False

# get /api/databases fills the database dropdown in React.
# A database shows up only if it is in PostgreSQL AND has a graph built in Neo4j.
@app.get("/api/databases", response_model=list[DatabaseInfo])
def databases():
    if DEMO_MODE:
        # reads from the team's sample file
        sample_file = (
                Path(__file__).resolve().parents[2]
                / "contracts" / "examples" / "metadata.adventureworks.sample.json"
        )
        with sample_file.open(encoding="utf-8") as file:
            sample = json.load(file)
        return [{"db_id": sample["db_id"], "table_count": len(sample["tables"]),
                 "schema_count": len(sample["schemas"])}]
    found = []
    for db_id in db.list_databases():
        tables = queries.list_tables(db_id)
        if not tables:
            continue   # no graph built for this database yet
        schemas = set()
        for table in tables:
            schemas.add(table.display_name.split(".")[0])   # "sales.customer" -> "sales"
    found.append({"db_id": db_id, "table_count": len(tables), "schema_count": len(schemas)})
    return found


# post /api/plan finds the tables and join path for a question without the LLM. Does not have a demo mode as it uses Neo4j.
@app.post("/api/plan", response_model=PlanResponse)
def plan(request: AskRequest):
    check_question(request)
    check_db_id(request.db_id)
    top_k = get_config().pipeline.top_k
    if request.options is not None and request.options.top_k is not None:
        top_k = request.options.top_k
    context, seed_tables = build_context(request.db_id, request.question, request.mode, top_k)
    return PlanResponse(seed_tables=seed_tables, join_plan=context.join_plan)


@app.get("/api/schema/{db_id}/tables", response_model=list[TableSummary])
def schema_tables(db_id: str):
    check_db_id(db_id)
    return queries.list_tables(db_id)
@app.get("/api/schema/{db_id}/tables/{table_uid}", response_model=TableWithRelationships)
def schema_table(db_id: str, table_uid: str):
    check_db_id(db_id)
    return queries.get_table_detail(db_id, table_uid)

