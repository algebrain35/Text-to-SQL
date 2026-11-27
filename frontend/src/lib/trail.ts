//Organizes the SQL join relationships for display as a chain or a list
import type { ExplainJoin, ExplainTable } from "../api/client";

export type Leg = ExplainJoin & { fromTable: ExplainTable; toTable: ExplainTable };

export type Trail =
  //Displays joins as a continuous chain of tables
  | { kind: "chain"; stops: ExplainTable[]; legs: Leg[] }
  //Displays joins individually when a continous chain cannot be formed
  | { kind: "list"; joins: ExplainJoin[] };

//Find the table a column belongs to using the longest matching table name
export function tableOfColumn(column: string, tables: ExplainTable[]): ExplainTable | undefined {
  let best: ExplainTable | undefined;
  for (const t of tables) {
    if (column.startsWith(`${t.display_name}.`) && (!best || t.display_name.length > best.display_name.length)) {
      best = t;
    }
  }
  return best;
}

//Organizes the joins into a chain or list and returns null if no joins exist
export function buildTrail(tables: ExplainTable[], joins: ExplainJoin[]): Trail | null {
  if (joins.length === 0) return null;

  //Identifies the tables connected by each join
  const legs: Leg[] = [];
  for (const join of joins) {
    const fromTable = tableOfColumn(join.from, tables);
    const toTable = tableOfColumn(join.to, tables);
    if (!fromTable || !toTable || fromTable.uid === toTable.uid) return { kind: "list", joins };
    legs.push({ ...join, fromTable, toTable });
  }

  //Creates a map of neighbouring tables
  const neighbours = new Map<string, string[]>();
  for (const leg of legs) {
    for (const [a, b] of [
      [leg.fromTable.uid, leg.toTable.uid],
      [leg.toTable.uid, leg.fromTable.uid],
    ] as const) {
      neighbours.set(a, [...(neighbours.get(a) ?? []), b]);
    }
  }

  //Checks whether the joins can form a single chain without brances or duplicate connections
  const isLine =
    legs.length === neighbours.size - 1 &&
    [...neighbours.values()].every((n) => n.length <= 2 && new Set(n).size === n.length);
  if (!isLine) return { kind: "list", joins };

  //Finds the starting table using the oringinal table order
  const order = new Map(tables.map((t, i) => [t.uid, i]));
  const ends = [...neighbours].filter(([, n]) => n.length === 1).map(([uid]) => uid);
  ends.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
  const byUid = new Map<string, ExplainTable>();
  for (const leg of legs) {
    byUid.set(leg.fromTable.uid, leg.fromTable);
    byUid.set(leg.toTable.uid, leg.toTable);
  }

  const stops: ExplainTable[] = [];
  const ordered: Leg[] = [];
  let current = ends[0] as string;
  let previous: string | null = null;

  //Traverses through the connected tables and orders their joins
  for (; ;) {
    stops.push(byUid.get(current) as ExplainTable);
    const next = (neighbours.get(current) as string[]).find((n) => n !== previous);
    if (next === undefined) break;
    ordered.push(
      legs.find(
        (l) =>
          (l.fromTable.uid === current && l.toTable.uid === next) ||
          (l.toTable.uid === current && l.fromTable.uid === next),
      ) as Leg,
    );
    previous = current;
    current = next;
  }

  //Uses a list if the chain does not reach every joined table
  if (stops.length !== neighbours.size) return { kind: "list", joins };
  return { kind: "chain", stops, legs: ordered };
}
