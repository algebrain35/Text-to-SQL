import { useEffect, useRef, useState, type SubmitEvent } from "react";
import { USE_MOCKS, ask, getDatabases, getHealth, plan, type DatabaseInfo, type Health, type Mode, type PlanResponse } from "./api/client";
import { Answer } from "./components/Answer";
import { DatabaseSelector } from "./components/DatabaseSelector";
import { JoinPlanSummary } from "./components/DeveloperPanel";
import { HealthBadge } from "./components/HealthBadge";
import { ModeSelector } from "./components/ModeSelector";
import { ErrorNotice } from "./components/Notices";
import { QuestionInput } from "./components/QuestionInput";
import { SchemaBrowser } from "./components/SchemaBrowser";
import type { AskResponse } from "./api/client";
import { readSetting, writeSetting } from "./lib/storage";

type View =
  | { kind: "idle" }
  | { kind: "loading"; what: "ask" | "plan" }
  | { kind: "answer"; response: AskResponse }
  | { kind: "plan"; result: PlanResponse }
  | { kind: "error"; error: unknown };

//Check wheter and error was caused by cancelling a request
function isAbort(error: unknown): boolean {
  if (error instanceof DOMException) {
    if (error.name === "AbortError") {
      return true;
    }
  }
  return false;
}

export default function App() {

  //Store available databases and track the FastAPI backend connection status
  const [databases, setDatabases] = useState<DatabaseInfo[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);

  //Store the user's selected database, question and query mode
  const [dbId, setDbId] = useState("");
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("graph");

  //Tracks whether developer mode is enabled
  const [developer, setDeveloper] = useState(() => {

    //Read the parameters from the browser URL
    const urlParameters = new URLSearchParams(window.location.search);

    //Check if developer mode was enabled by user on URL
    const developerParameter = urlParameters.get("dev");

    if (developerParameter === "1") {
      return true;
    }

    //Check if developer mode was previously saved
    const savedDeveloper = readSetting("developer");

    if (savedDeveloper == "true") {
      return true;
    }

    //Developer mode is disabled by default
    return false;
  });

  //Tracks the current frontend state, possible states are (idle, loading, answer, plan or error)
  const [view, setView] = useState<View>({ kind: "idle" });

  //Tracks the currently active API request, so that it can be cancelled when a new request starts
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    //Check the FastAPI backend's connection status
    getHealth(controller.signal)

      //If successful, store the backend health status in React state
      .then(setHealth)

      //If the request Fails (not due to cancelling the request) the backend health check failed
      .catch((error) => !isAbort(error) && setHealthFailed(true));

    //Request the list of available databases from FastAPI  
    getDatabases(controller.signal)
      .then((list) => {
        setDatabases(list);

        //Retrieve the previously selected databse
        const saved = readSetting("database");

        let selectedDatabase = "";

        //Search the returned databases for the previously saved ID
        const matchingDatabase = list.find((database) => {
          return database.db_id === saved;
        });

        //if the saved database still exists select it
        if (matchingDatabase !== undefined) {
          selectedDatabase = matchingDatabase.db_id;
        }

        //Otherwise select the first available database
        else if (list.length > 0) {
          selectedDatabase = list[0].db_id;
        }

        //Update the frontend with the selected database
        setDbId(selectedDatabase);

      })
      .catch((error) => {

        //Ignores errors when cancelling requests
        if (isAbort(error)) {
          return;
        }

        //If the retrieval of the database fails, empty list used and error shown
        setDatabases([]);
        setView({
          kind: "error",
          error
        });
      });
    return () => controller.abort();
  }, []);

  //Handles database selction when user changes databases
  function chooseDatabase(id: string) {

    //update the selected database in react state
    setDbId(id);

    //Save the selected database for future visit
    writeSetting("database", id);

    //Clear previous results to avoid displayin results from another database
    setView({ kind: "idle" });
  }

  //Handles developer mode
  function toggleDeveloper(on: boolean) {
    setDeveloper(on);
    writeSetting("developer", String(on));
    if (!on) setMode("graph");
  }

  //Handles SQL queries and join planning
  async function run(what: "ask" | "plan") {

    //Cancel any previous request still running
    if (inFlight.current !== null) {
      inFlight.current.abort();
    }

    //Create a controller for the new request
    const controller = new AbortController();
    inFlight.current = controller;

    //Update the frontend to show loading status
    setView({ kind: "loading", what });

    //Determine which mode should be used
    let requestMode: Mode = "graph";

    if (developer == true) {
      requestMode = mode;
    }

    //Prepare the request to send to FastAPI
    const request = {
      question: question.trim(),
      db_id: dbId,
      mode: requestMode
    };


    try {

      //Run the Full SQL generation request
      if (what === "ask") {

        const response = await ask(request, controller.signal);

        setView({
          kind: "answer",
          response: response
        });
      }
      //Run only the planning operation
      else {

        const result = await plan(request, controller.signal);

        setView({
          kind: "plan",
          result: result
        });
      }
    } catch (error) {

      //Ignore errors caused by intentionally cancelling requests
      if (!isAbort(error)) {

        setView({
          kind: "error",
          error: error

        });
      }
    }
  }

  //Handles submitting a question to the backend
  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();

    const hasQuestion = question.trim() !== "";
    const hasDatabase = dbId !== "";

    //Only submit if both user has entered a question and selected a database
    if (hasQuestion && hasDatabase) {
      void run("ask");
    }
  }

  const busy = view.kind === "loading";

  //Determine which loading message to display
  let loadingMessage = "";

  if (view.kind === "loading") {

    if (view.what === "ask") {
      loadingMessage = "Writing and running the query";
    }
    else {
      loadingMessage = "Planning the join";
    }
  }

  //Determine whether the Plan Only button should be disabled
  let planDisabled = false;

  if (busy) {
    planDisabled = true;
  }
  else if (question.trim() === "") {
    planDisabled = true;
  }
  else if (dbId === "") {
    planDisabled = true;
  }

  //Default to displaying nothing while the frontend is loading
  let resultDisplay = <></>;

  //Display instructions before a question is submitted
  if (view.kind === "idle") {
    resultDisplay = (
      <p className="empty">
        Type in below a question you want to use to query the database in plain English.
        You will get: the SQL, the result, and which tables were used and how they connect.
      </p>
    );
  }

  //Display errors if the request fails
  else if (view.kind === "error") {
    resultDisplay = <ErrorNotice error={view.error} />;
  }

  //Display the generated SQL and query results
  else if (view.kind === "answer") {
    resultDisplay = <Answer response={view.response} developer={developer} />;
  }

  //Display the database join plan if planning was requested
  else if (view.kind === "plan") {
    resultDisplay = (
      <section className="answer" aria-labelledby="plan-heading">
        <h2 id="plan-heading">Plan (no model call)</h2>
        <JoinPlanSummary
          plan={view.result.join_plan}
          seeds={view.result.seed_tables}
        />
      </section>
    );
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" className="logo">
            <path d="M12 3v18" />
            <circle cx="12" cy="5" r="3" className="seed" />
            <circle cx="12" cy="12" r="2.6" className="bridge" />
            <circle cx="12" cy="19" r="3" className="seed" />
          </svg>
          <h1>SchemaGraph</h1>
        </div>
        <HealthBadge health={health} failed={healthFailed} />
        <label className="switch">
          <input type="checkbox" checked={developer} onChange={(event) => toggleDeveloper(event.target.checked)} />
          Developer view
        </label>
      </header>

      {USE_MOCKS && <p className="banner">Showing sample data. Nothing is sent to the API.</p>}

      <main>
        <form onSubmit={submit} className="ask-form">
          <DatabaseSelector databases={databases} value={dbId} onChange={chooseDatabase} disabled={busy} />
          <QuestionInput value={question} onChange={setQuestion} busy={busy} disabled={!dbId} />
          {developer && (
            <div className="dev-controls">
              <ModeSelector value={mode} onChange={setMode} disabled={busy} />
              <button type="button"
                className="quiet"
                disabled={planDisabled}
                onClick={() => void run("plan")}>
                Plan only (no model call)
              </button>
            </div>
          )}
        </form>

        <div className="status-line" role="status" aria-live="polite">
          {loadingMessage}
        </div>

        {resultDisplay}

        <SchemaBrowser dbId={dbId} />
      </main>
    </div>
  );
}
