import { formatSql } from "../lib/sql";
import type { components } from "../api/types";

type Attempt = components["schemas"]["Attempt"];

/** Every SQL the model wrote, in order, with the database's complaint if it failed. Collapsed by default (SPEC 5.22). */
export function AttemptsView({ attempts }: { attempts: Attempt[] }) {
  if (attempts.length === 0) return null;
  const failed = attempts.filter((a) => a.error).length;
  return (
    <details className="attempts">
      <summary>
        Attempts ({attempts.length}
        {failed > 0 ? `, ${failed} failed` : ""})
      </summary>
      <ol>
        {attempts.map((attempt, i) => (
          <li key={i}>
            <pre className="code">
              <code>{formatSql(attempt.sql)}</code>
            </pre>
            {attempt.error ? <p className="error-text">{attempt.error}</p> : <p className="hint">Ran without an error.</p>}
          </li>
        ))}
      </ol>
    </details>
  );
}
