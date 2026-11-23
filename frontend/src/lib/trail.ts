import type { ExplainJoin, ExplainTable } from "../api/client";

export type Leg = ExplainJoin & { fromTable: ExplainTable; toTable: ExplainTable };

export type Trail =
  /** The joins form one line of tables: stops[i] and stops[i + 1] are joined by legs[i]. */
  | { kind: "chain"; stops: ExplainTable[]; legs: Leg[] }
  /** The joins branch, repeat or name a table we cannot place: show them one by one. */
  | { kind: "list"; joins: ExplainJoin[] };

/** The table a column belongs to. `column` is a display name (`schema.table.column`); the owner is the longest
 *  known table name it starts with, so names containing dots are never cut apart. */
export function tableOfColumn(column: string, tables: ExplainTable[]): ExplainTable | undefined {
  let best: ExplainTable | undefined;
  for (const t of tables) {
    if (column.startsWith(`${t.display_name}.`) && (!best || t.display_name.length > best.display_name.length)) {
      best = t;
    }
  }
  return best;
}

/** Arrange an explanation's join path for drawing, or null when there are no joins. */
export function buildTrail(tables: ExplainTable[], joins: ExplainJoin[]): Trail | null {
  if (joins.length === 0) return null;

  const legs: Leg[] = [];
  for (const join of joins) {
    const fromTable = tableOfColumn(join.from, tables);
    const toTable = tableOfColumn(join.to, tables);
    if (!fromTable || !toTable || fromTable.uid === toTable.uid) return { kind: "list", joins };
    legs.push({ ...join, fromTable, toTable });
  }

  const neighbours = new Map<string, string[]>();
  for (const leg of legs) {
    for (const [a, b] of [
      [leg.fromTable.uid, leg.toTable.uid],
      [leg.toTable.uid, leg.fromTable.uid],
    ] as const) {
      neighbours.set(a, [...(neighbours.get(a) ?? []), b]);
    }
  }
  const isLine =
    legs.length === neighbours.size - 1 &&
    [...neighbours.values()].every((n) => n.length <= 2 && new Set(n).size === n.length);
  if (!isLine) return { kind: "list", joins };

  // Start from the line's end that the explanation lists first, then walk to the other end.
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
  for (;;) {
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
  // A line that does not reach every joined table means the joins were not one connected line.
  if (stops.length !== neighbours.size) return { kind: "list", joins };
  return { kind: "chain", stops, legs: ordered };
}
