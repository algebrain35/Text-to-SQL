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

    # For the final team backend, Mohammed should also add:
    # - errors for unavailable PostgreSQL / Neo4j / LLM (HTTP 503),
    # - the GET /api/health and GET /api/databases routes from SPEC.md,
    # - any other planned routes and security/access controls.
