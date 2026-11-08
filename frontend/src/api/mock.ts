/**
 * Answers for VITE_USE_MOCKS=true, built from copies of contracts/examples (a test keeps the copies in sync).
 * Questions starting with "mock:" show a failure: invalid_sql, no_schema, bad (HTTP 400) or unavailable (HTTP 503).
 */
import { displayName } from "../lib/names";
import type { components } from "./types";
import askSample from "../mocks/ask_response.json";
import joinPlanSample from "../mocks/join_plan.json";
import metadataSample from "../mocks/metadata.adventureworks.json";

type Schemas = components["schemas"];
type AskRequest = Omit<Schemas["AskRequest"], "mode"> & { mode?: Schemas["AskRequest"]["mode"] };
type AskResponse = Schemas["AskResponse"];

const ASK = askSample as unknown as AskResponse;
const PLAN = joinPlanSample as unknown as Schemas["JoinPlan"];
const META = metadataSample as unknown as {
  db_id: string;
  schemas: unknown[];
  tables: Schemas["TableMeta"][];
  columns: Schemas["ColumnMeta"][];
  relationships: Schemas["RelationshipMeta"][];
};

let latencyMs = 450;

/** Tests set this to 0 so the mock answers at once. */
export function setMockLatency(ms: number): void {
  latencyMs = ms;
}

export class MockHttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
  }
}

function wait(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, latencyMs);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
}

export async function getHealth(): Promise<Schemas["Health"]> {
  return { postgres: "ok", neo4j: "ok", llm: "ok" };
}

export async function getDatabases(): Promise<Schemas["DatabaseInfo"][]> {
  return [{ db_id: META.db_id, table_count: META.tables.length, schema_count: META.schemas.length }];
}

export async function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  await wait(signal);
  const base: AskResponse = { ...ASK, question: req.question, db_id: req.db_id, mode: req.mode ?? "graph" };
  const trigger = /^mock:(\w+)/.exec(req.question.trim())?.[1];
  switch (trigger) {
    case "bad":
      throw new MockHttpError(400, "question: String should have at least 1 character");
    case "unavailable":
      throw new MockHttpError(503, "Neo4j is unreachable or refused the request.");
    case "no_schema":
      return {
        ...base,
        status: "no_schema",
        sql: null,
        result: null,
        explanation: { summary: "No table in this database looked related to the question.", tables: [], join_path: [], warnings: [] },
        join_plan: null,
        attempts: [],
      };
    case "invalid_sql":
      return {
        ...base,
        status: "invalid_sql",
        result: null,
        metrics: { ...ASK.metrics!, attempt_count: 3 },
        sql: "SELECT firstname FROM person.persons",
        explanation: { ...ASK.explanation!, warnings: ["The query used a table that does not exist."] },
        attempts: [
          { sql: "SELECT firstname FROM person.persons", error: 'relation "person.persons" does not exist' },
          { sql: "SELECT firstname FROM person.persons p", error: 'relation "person.persons" does not exist' },
          { sql: "SELECT firstname FROM person.person_names", error: 'relation "person.person_names" does not exist' },
        ],
      };
    default:
      return base;
  }
}

export async function plan(req: AskRequest, signal?: AbortSignal): Promise<Schemas["PlanResponse"]> {
  await wait(signal);
  const graph = (req.mode ?? "graph") === "graph";
  return { seed_tables: PLAN.seed_tables ?? [], join_plan: graph ? PLAN : null };
}

export async function listTables(): Promise<Schemas["TableSummary"][]> {
  return META.tables.map((t) => ({
    uid: t.uid,
    display_name: displayName(t.uid),
    description: t.description ?? null,
    column_count: META.columns.filter((c) => c.table_uid === t.uid).length,
  }));
}

export async function getTable(uid: string): Promise<Schemas["TableWithRelationships"]> {
  const table = META.tables.find((t) => t.uid === uid);
  if (!table) throw new MockHttpError(404, `No table ${displayName(uid)} in the graph.`);
  const columns = META.columns.filter((c) => c.table_uid === table.uid);
  const tableOf = new Map(META.columns.map((c) => [c.uid, c.table_uid]));
  return {
    table,
    columns,
    references: META.relationships.filter((r) => tableOf.get(r.from_column) === uid),
    referenced_by: META.relationships.filter((r) => tableOf.get(r.to_column) === uid),
  };
}
