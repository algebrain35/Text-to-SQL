import { MODES, type Mode } from "../api/client";

const LABELS: Record<Mode, { name: string; detail: string }> = {
  graph: { name: "Graph", detail: "matched tables plus the tables that join them" },
  retrieval: { name: "Retrieval", detail: "matched tables only" },
  full: { name: "Full", detail: "every table in the database" },
};

type Props = { value: Mode; onChange: (mode: Mode) => void; disabled?: boolean };

/** Developer view only (SPEC 5.22): how much of the schema the model is shown. */
export function ModeSelector({ value, onChange, disabled }: Props) {
  return (
    <fieldset className="modes" disabled={disabled}>
      <legend>Schema mode</legend>
      {MODES.map((mode) => (
        <label key={mode} className="mode">
          <input type="radio" name="mode" value={mode} checked={value === mode} onChange={() => onChange(mode)} />
          <span>
            <strong>{LABELS[mode].name}</strong>: {LABELS[mode].detail}
          </span>
        </label>
      ))}
    </fieldset>
  );
}
