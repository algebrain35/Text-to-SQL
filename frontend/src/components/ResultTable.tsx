import type { components } from "../api/types";

type QueryResult = components["schemas"]["QueryResult"];

const NUMBER = /^-?\d+(\.\d+)?$/;

export function csvOf(result: QueryResult): string {
  const cell = (value: string | null) => (value === null ? "" : /[",\n\r]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value);
  return [result.columns, ...result.rows].map((row) => row.map((v) => cell(v as string | null)).join(",")).join("\r\n");
}

function download(result: QueryResult) {
  const url = URL.createObjectURL(new Blob([csvOf(result)], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "result.csv";
  link.click();
  URL.revokeObjectURL(url);
}

/** A column is numeric when every value in it is a number (or null), so its header can line up with its values. */
export function numericColumns(result: QueryResult): boolean[] {
  return result.columns.map((_, c) => {
    const values = result.rows.map((row) => row[c]).filter((v) => v !== null && v !== undefined);
    return values.length > 0 && values.every((v) => NUMBER.test(String(v)));
  });
}

export function ResultTable({ result }: { result: QueryResult }) {
  const count = result.rows.length;
  const numeric = numericColumns(result);
  return (
    <section aria-labelledby="result-heading">
      <div className="section-head">
        <h3 id="result-heading">Result</h3>
        <span className="hint">{count === 1 ? "1 row" : `${count} rows`}</span>
        {count > 0 && (
          <button type="button" className="quiet" onClick={() => download(result)}>
            Download CSV
          </button>
        )}
      </div>
      {count === 0 ? (
        <p className="empty">The query ran and returned no rows.</p>
      ) : (
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Query result rows">
          <table>
            <thead>
              <tr>
                {result.columns.map((name, c) => (
                  <th key={name} scope="col" className={numeric[c] ? "num" : undefined}>
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, r) => (
                <tr key={r}>
                  {row.map((value, c) => (
                    <td key={c} className={numeric[c] ? "num" : undefined}>
                      {value === null || value === undefined ? <span className="null">null</span> : value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.truncated && (
        <p className="note">
          Showing the first {count} rows. The query returned more; add a filter or a limit to your question to narrow it.
        </p>
      )}
    </section>
  );
}
