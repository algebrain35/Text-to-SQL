import type { AskResponse, JoinPlan, PlanResponse } from "../api/client";
import { displayName } from "../lib/names";

const STAGES = ["retrieval", "planning", "generation", "validation", "execution", "explanation"] as const;

export function JoinPlanSummary({ plan, seeds }: { plan: JoinPlan | null | undefined; seeds?: PlanResponse["seed_tables"] }) {
  const seedList = seeds ?? plan?.seed_tables ?? [];
  return (
    <div className="dev-block">
      {seedList.length > 0 && (
        <>
          <h4>Retrieved tables</h4>
          <table className="plain">
            <thead>
              <tr>
                <th scope="col">Table</th>
                <th scope="col">Score</th>
                <th scope="col">Matched on</th>
              </tr>
            </thead>
            <tbody>
              {seedList.map((seed) => (
                <tr key={seed.uid}>
                  <td className="ident">{displayName(seed.uid)}</td>
                  <td className="num">{seed.score.toFixed(3)}</td>
                  <td>{seed.matched_on}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {plan ? (
        <>
          <h4>Join plan</h4>
          <dl>
            <dt>Status</dt>
            <dd>{plan.status}</dd>
            <dt>Algorithm</dt>
            <dd>{plan.algorithm}</dd>
            <dt>Total cost</dt>
            <dd>{plan.total_cost}</dd>
            {(plan.unreachable_seeds ?? []).length > 0 && (
              <>
                <dt>Unreachable</dt>
                <dd className="ident">{(plan.unreachable_seeds ?? []).map(displayName).join(", ")}</dd>
              </>
            )}
            {(plan.dropped_seeds ?? []).length > 0 && (
              <>
                <dt>Dropped</dt>
                <dd className="ident">{(plan.dropped_seeds ?? []).map(displayName).join(", ")}</dd>
              </>
            )}
          </dl>
          {(plan.joins ?? []).length > 0 && (
            <ul className="plain">
              {(plan.joins ?? []).map((join, i) => (
                <li key={i}>
                  <span className="ident small">
                    {displayName(join.from_column)} = {displayName(join.to_column)}
                  </span>{" "}
                  {join.source}, confidence {join.confidence.toFixed(2)}, {join.cardinality},{" "}
                  {join.traversal === "one_to_many" ? "one to many (can repeat rows)" : "many to one"}
                </li>
              ))}
            </ul>
          )}
          {(plan.alternatives ?? []).length > 0 && (
            <>
              <h4>Other routes considered</h4>
              <ul className="plain">
                {(plan.alternatives ?? []).map((alt, i) => (
                  <li key={i}>
                    <span className="ident small">{alt.tables.map(displayName).join(", ")}</span> (cost {alt.total_cost})
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      ) : (
        <p className="hint">No join plan: this mode does not use the graph.</p>
      )}
    </div>
  );
}

/** Developer view: what the pipeline did to produce this answer. */
export function DeveloperPanel({ response }: { response: AskResponse }) {
  const m = response.metrics;
  return (
    <section aria-labelledby="dev-heading" className="developer">
      <h3 id="dev-heading">Developer details</h3>
      <dl>
        <dt>Request</dt>
        <dd className="ident small">{response.request_id}</dd>
        <dt>Mode</dt>
        <dd>{response.mode}</dd>
        {m && (
          <>
            <dt>Tokens</dt>
            <dd>
              {m.input_tokens} in, {m.output_tokens} out
            </dd>
            <dt>Attempts</dt>
            <dd>{m.attempt_count}</dd>
            <dt>Time</dt>
            <dd>
              {m.latency_ms
                ? [...STAGES.filter((s) => m.latency_ms![s]).map((s) => `${s} ${m.latency_ms![s]} ms`), `total ${m.latency_ms.total} ms`].join(", ")
                : "not recorded"}
            </dd>
          </>
        )}
      </dl>
      <JoinPlanSummary plan={response.join_plan} />
    </section>
  );
}
