import { useState } from "react";
import { formatSql } from "../lib/sql";

export function SqlView({ sql }: { sql: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      //If copying to clipboard fails, users can still manually select the SQL
      setCopied(false);
    }
  }

  return (
    <section aria-labelledby="sql-heading">
      <div className="section-head">
        <h3 id="sql-heading">SQL</h3>
        <button type="button" className="quiet" onClick={copy}>
          Copy SQL
        </button>
        <span className="hint" role="status">
          {copied ? "Copied" : ""}
        </span>
      </div>
      <pre className="code" tabIndex={0}>
        <code>{formatSql(sql)}</code>
      </pre>
    </section>
  );
}
