import type { ExplainTable } from "../api/client";

export const ROLE_LABEL: Record<ExplainTable["role"], string> = {
  seed: "Matches your question",
  bridge: "Connects other tables",
  context: "Included as context",
};

/** A table's role as a shape, so it reads without colour: filled dot = matched, ring = connects, square ring = context. */
export function Marker({ role }: { role: ExplainTable["role"] }) {
  return <span className={`marker ${role}`} aria-hidden="true" />;
}
