import type { Health } from "../api/client";

const NAMES: [keyof Health, string][] = [
  ["postgres", "Database"],
  ["neo4j", "Graph"],
  ["llm", "Model"],
];

export function HealthBadge({ health, failed }: { health: Health | null; failed: boolean }) {
  if (failed) return <p className="health bad" role="status">API not reachable</p>;
  if (!health) return <p className="health" role="status">Checking services</p>;
  const problems = NAMES.filter(([key]) => health[key] !== "ok");
  if (problems.length === 0) return <p className="health ok" role="status">Services ready</p>;
  return (
    <p className="health bad" role="status">
      {problems.map(([key, name]) => `${name}: ${health[key] === "not_configured" ? "not configured" : health[key]}`).join(", ")}
    </p>
  );
}
