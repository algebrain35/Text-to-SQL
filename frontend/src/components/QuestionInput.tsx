import type { KeyboardEvent } from "react";

type Props = {
  value: string;
  onChange: (question: string) => void;
  busy: boolean;
  disabled?: boolean;
};

/** The question box and its button. The parent form handles submitting; Ctrl or Cmd+Enter submits from the box. */
export function QuestionInput({ value, onChange, busy, disabled }: Props) {
  const empty = value.trim() === "";

  function submitOnShortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (!empty && !busy && !disabled) event.currentTarget.form?.requestSubmit();
    }
  }

  return (
    <div className="question">
      <label htmlFor="question">Your question</label>
      <textarea
        id="question"
        rows={3}
        value={value}
        placeholder="For example: What is the total order value for each customer?"
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={submitOnShortcut}
        aria-describedby="question-hint"
      />
      <div className="question-actions">
        <button type="submit" className="primary" disabled={empty || busy || disabled}>
          {busy ? "Asking" : "Ask"}
        </button>
        <span id="question-hint" className="hint">
          Ctrl or Cmd + Enter also asks.
        </span>
      </div>
    </div>
  );
}
