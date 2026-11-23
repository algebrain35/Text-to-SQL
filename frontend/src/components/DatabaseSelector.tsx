import type { DatabaseInfo } from "../api/client";

type Props = {
  databases: DatabaseInfo[] | null;
  value: string;
  onChange: (dbId: string) => void;
  disabled?: boolean;
};

export function DatabaseSelector({ databases, value, onChange, disabled }: Props) {
  return (
    <div className="field">
      <label htmlFor="database">Database</label>
      <select
        id="database"
        value={value}
        disabled={disabled || databases === null || databases.length === 0}
        onChange={(event) => onChange(event.target.value)}
      >
        {databases === null && <option value="">Loading databases</option>}
        {databases?.length === 0 && <option value="">No databases found</option>}
        {databases?.map((db) => (
          <option key={db.db_id} value={db.db_id}>
            {db.db_id} ({db.table_count} {db.table_count === 1 ? "table" : "tables"})
          </option>
        ))}
      </select>
    </div>
  );
}
