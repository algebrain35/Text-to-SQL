import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  USE_MOCKS,
  ask,
  getDatabases,
  getHealth,
  plan,
  type DatabaseInfo,
  type Health,
  type Mode,
  type PlanResponse,
} from "./api/client";
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

const isAbort = (error: unknown) => error instanceof DOMException && error.name === "AbortError";

export default function App() {
  const [databases, setDatabases] = useState<DatabaseInfo[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthFailed, setHealthFailed] = useState(false);
  const [dbId, setDbId] = useState("");
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("graph");
  const [developer, setDeveloper] = useState(
    () => new URLSearchParams(window.location.search).get("dev") === "1" || readSetting("developer") === "true",
  );
  const [view, setView] = useState<View>({ kind: "idle" });
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then(setHealth)
      .catch((error) => !isAbort(error) && setHealthFailed(true));
    getDatabases(controller.signal)
      .then((list) => {
        setDatabases(list);
        const saved = readSetting("database");
        setDbId(list.find((d) => d.db_id === saved)?.db_id ?? list[0]?.db_id ?? "");
      })
      .catch((error) => {
        if (isAbort(error)) return;
        setDatabases([]);
        setView({ kind: "error", error });
      });
    return () => controller.abort();
  }, []);

  function chooseDatabase(id: string) {
    setDbId(id);
    writeSetting("database", id);
    setView({ kind: "idle" }); // an answer about another database would mislead
  }

  function toggleDeveloper(on: boolean) {
    setDeveloper(on);
    writeSetting("developer", String(on));
    if (!on) setMode("graph");
  }

  async function run(what: "ask" | "plan") {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setView({ kind: "loading", what });
    const request = { question: question.trim(), db_id: dbId, mode: developer ? mode : ("graph" as const) };
    try {
      if (what === "ask") setView({ kind: "answer", response: await ask(request, controller.signal) });
      else setView({ kind: "plan", result: await plan(request, controller.signal) });
    } catch (error) {
      if (!isAbort(error)) setView({ kind: "error", error });
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (question.trim() && dbId) void run("ask");
  }

  const busy = view.kind === "loading";

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
              <button type="button" className="quiet" disabled={busy || !question.trim() || !dbId} onClick={() => void run("plan")}>
                Plan only (no model call)
              </button>
            </div>
          )}
        </form>

        <div className="status-line" role="status" aria-live="polite">
          {view.kind === "loading" && (view.what === "ask" ? "Writing and running the query" : "Planning the join")}
        </div>

        {view.kind === "idle" && (
          <p className="empty">
            Ask a question in plain English. You get the SQL, the result, and which tables were used and how they connect.
          </p>
        )}
        {view.kind === "error" && <ErrorNotice error={view.error} />}
        {view.kind === "answer" && <Answer response={view.response} developer={developer} />}
        {view.kind === "plan" && (
          <section className="answer" aria-labelledby="plan-heading">
            <h2 id="plan-heading">Plan (no model call)</h2>
            <JoinPlanSummary plan={view.result.join_plan} seeds={view.result.seed_tables} />
          </section>
        )}

        <SchemaBrowser dbId={dbId} />
      </main>
    </div>
  );
}
