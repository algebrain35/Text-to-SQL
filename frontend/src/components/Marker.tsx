import type { ExplainTable } from "../api/client";

export const ROLE_LABEL: Record<ExplainTable["role"], string> = {
  seed: "Matches your question",
  bridge: "Connects other tables",
  context: "Included as context",
};

//Displays table roles using shapes: filled dot for matched, ring for connecting, and square ring for context
export function Marker({ role }: { role: ExplainTable["role"] }) {
  return <span className={`marker ${role}`} aria-hidden="true" />;
}
