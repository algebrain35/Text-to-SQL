import { useEffect, useRef } from "react";
import type { AskResponse } from "../api/client";
import { displayName } from "../lib/names";
import { AttemptsView } from "./AttemptsView";
import { DeveloperPanel } from "./DeveloperPanel";
import { ExplanationPanel } from "./ExplanationPanel";
import { StatusNotice } from "./Notices";
import { JoinPathView } from "./JoinPathView";
import { ResultTable } from "./ResultTable";
import { SqlView } from "./SqlView";
import { TablesUsed } from "./TablesUsed";
import { WarningsView } from "./WarningsView";

/** Warnings the page can work out from the response itself, on top of the ones the explanation lists. */
export function derivedWarnings(response: AskResponse): string[] {
  const out: string[] = [];
  const plan = response.join_plan;
  const unreachable = plan?.unreachable_seeds ?? [];
  if (plan?.status === "partial" || unreachable.length > 0) {
    out.push(
      `Some matched tables could not be connected to the others${unreachable.length ? ` (${unreachable.map(displayName).join(", ")})` : ""}, so the query may be missing a join.`,
    );
  }
  const dropped = plan?.dropped_seeds ?? [];
  if (dropped.length > 0) {
    out.push(`Left out because they were too far from the other tables: ${dropped.map(displayName).join(", ")}.`);
  }
  if ((response.explanation?.join_path ?? []).some((j) => j.source === "inferred")) {
    out.push("This answer relies on a join the database does not declare. Check that it matches what you meant.");
  }
  return out;
}

export function Answer({ response, developer }: { response: AskResponse; developer: boolean }) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), [response]);

  const explanation = response.explanation;
  const warnings = [...(explanation?.warnings ?? []), ...derivedWarnings(response)];

  return (
    <article className="answer" aria-labelledby="answer-heading">
      <h2 id="answer-heading" ref={heading} tabIndex={-1}>
        {response.status === "success" ? "Answer" : "No answer"}
      </h2>
      <StatusNotice response={response} />
      {explanation && <ExplanationPanel explanation={explanation} />}
      {explanation && <JoinPathView tables={explanation.tables ?? []} joins={explanation.join_path ?? []} plan={response.join_plan} />}
      {response.result && <ResultTable result={response.result} />}
      {response.sql && <SqlView sql={response.sql} />}
      {explanation && <TablesUsed tables={explanation.tables ?? []} />}
      <WarningsView warnings={warnings} />
      <AttemptsView attempts={response.attempts ?? []} />
      {developer && <DeveloperPanel response={response} />}
    </article>
  );
}
