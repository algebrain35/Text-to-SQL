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

//Generates additional warnings based on the API response
export function derivedWarnings(response: AskResponse): string[] {
  const out: string[] = [];
  const plan = response.join_plan;
  const unreachable = plan?.unreachable_seeds ?? [];

  //Determine which unreachable tables to include in the warning
  let unreachableText = "";

  if (unreachable.length > 0) {
    unreachableText = ` (${unreachable.map(displayName).join(", ")})`;
  }

  if (plan?.status === "partial" || unreachable.length > 0) {
    out.push(
      `Some matched tables could not be connected to the others${unreachableText}, so the query may be missing a join.`,
    );
  }

  const dropped = plan?.dropped_seeds ?? [];
  if (dropped.length > 0) {
    out.push(`Left out because they were too far from the other tables: ${dropped.map(displayName).join(", ")}.`);
  }

  //Check if any database joins were inferred
  const joinPath = response.explanation?.join_path ?? [];

  const hasInferredJoin = joinPath.some((join) => {
    return join.source === "inferred";
  });

  if (hasInferredJoin) {
    out.push("This answer relies on a join the database does not declare. Check that it matches what you meant.");
  }

  return out;
}

export function Answer({ response, developer }: { response: AskResponse; developer: boolean }) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), [response]);

  const explanation = response.explanation;

  //Store the warnings that will be displayed to the user
  const warnings: string[] = [];

  //Add any warnings returned by FastAPI
  if (explanation?.warnings) {
    warnings.push(...explanation.warnings);
  }

  //Add any additional warnings generated from the response
  const additionalWarnings = derivedWarnings(response);
  warnings.push(...additionalWarnings);

  //Determine the heading based on the response status
  let headingText = "No answer";

  if (response.status === "success") {
    headingText = "Answer";
  }

  return (
    <article className="answer" aria-labelledby="answer-heading">
      <h2 id="answer-heading" ref={heading} tabIndex={-1}>
        {headingText}
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
