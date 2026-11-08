import type { ExplainTable } from "../api/client";
import { Marker, ROLE_LABEL } from "./Marker";

/** The tables the answer was built from. Matched tables and connecting tables look different (SPEC 5.22). */
export function TablesUsed({ tables }: { tables: ExplainTable[] }) {
  if (tables.length === 0) return null;
  return (
    <section aria-labelledby="tables-heading">
      <h3 id="tables-heading">Tables used</h3>
      <ul className="tables-used">
        {tables.map((table) => (
          <li key={table.uid}>
            <Marker role={table.role} />
            <div>
              <span className="ident">{table.display_name}</span>
              <span className={`role ${table.role}`}>{ROLE_LABEL[table.role]}</span>
              {table.reason && <p>{table.reason}</p>}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
