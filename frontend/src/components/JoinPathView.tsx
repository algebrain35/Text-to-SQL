import type { ExplainJoin, ExplainTable, JoinPlan } from "../api/client";
import { displayName } from "../lib/names";
import { buildTrail, type Leg } from "../lib/trail";
import { Marker } from "./Marker";

/** Confidence of the planner's join for a displayed one, when the plan is available. Joins are matched by their column names. */
function confidenceOf(plan: JoinPlan | null | undefined, join: ExplainJoin): number | undefined {
  return plan?.joins?.find((j) => displayName(j.from_column) === join.from && displayName(j.to_column) === join.to)
    ?.confidence;
}

function JoinNote({ join, plan }: { join: ExplainJoin; plan?: JoinPlan | null }) {
  const confidence = confidenceOf(plan, join);
  return (
    <>
      <p>{join.text}</p>
      <p className="ident small">
        {join.from} = {join.to}
      </p>
      {join.source === "inferred" && (
        <p className="inferred-note">
          Inferred from column names and data, not declared in the database
          {confidence !== undefined ? ` (confidence ${confidence.toFixed(2)})` : ""}.
        </p>
      )}
    </>
  );
}

function Chain({ stops, legs, plan }: { stops: ExplainTable[]; legs: Leg[]; plan?: JoinPlan | null }) {
  return (
    <ol className="trail">
      {stops.map((stop, i) => {
        const leg = legs[i];
        return [
          <li key={stop.uid} className={`stop ${stop.role}`}>
            <div className="rail">
              <Marker role={stop.role} />
              {leg && <span className="line" />}
            </div>
            <div className="stop-body">
              <span className="ident">{stop.display_name}</span>
            </div>
          </li>,
          leg && (
            <li key={`leg-${i}`} className={`leg ${leg.source}`}>
              <div className="rail">
                <span className="line" />
              </div>
              <div className="leg-body">
                <JoinNote join={leg} plan={plan} />
              </div>
            </li>
          ),
        ];
      })}
    </ol>
  );
}

/**
 * How the tables connect, drawn as a line: a filled stop for each table that matched the question, a ring for each
 * table passed through to connect them. A solid line is a declared join; a dashed amber line is an inferred one.
 */
export function JoinPathView({
  tables,
  joins,
  plan,
}: {
  tables: ExplainTable[];
  joins: ExplainJoin[];
  plan?: JoinPlan | null;
}) {
  const trail = buildTrail(tables, joins);
  if (!trail) return null;
  return (
    <section aria-labelledby="path-heading">
      <h3 id="path-heading">How the tables connect</h3>
      {trail.kind === "chain" ? (
        <Chain stops={trail.stops} legs={trail.legs} plan={plan} />
      ) : (
        <ul className="join-list">
          {trail.joins.map((join, i) => (
            <li key={i} className={join.source}>
              <JoinNote join={join} plan={plan} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
