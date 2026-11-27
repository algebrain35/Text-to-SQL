import type { KeyboardEvent } from "react";

type Props = {
  value: string;
  onChange: (question: string) => void;
  busy: boolean;
  disabled?: boolean;
};

//Handles the question input, Ask button and Ctrl/Cmd + Enter Shortcut
export function QuestionInput({ value, onChange, busy, disabled }: Props) {
  const empty = value.trim() === "";

  //Determine the button text based on whether a request is running
  let buttonText = "Ask";

  if (busy) {
    buttonText = "Asking";
  }

  //Determine whether the Ask button should be disabled
  let buttonDisabled = false;

  if (empty) {
    buttonDisabled = true;
  }
  else if (busy) {
    buttonDisabled = true;
  }
  else if (disabled) {
    buttonDisabled = true;
  }

  function submitOnShortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
    //Check if Enter was pressed
    const isEnter = event.key === "Enter";

    //Check if Ctrl or Cmd was pressed
    const shortcutKey = event.ctrlKey || event.metaKey;

    //Only activate the shortcut when both conditions are met
    if (isEnter && shortcutKey) {
      event.preventDefault();

      //Submit the question only if the Ask button is enabled
      if (!buttonDisabled) {
        event.currentTarget.form?.requestSubmit();
      }
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
        <button type="submit" className="primary" disabled={buttonDisabled}>
          {buttonText}
        </button>
        <span id="question-hint" className="hint">
          Ctrl or Cmd + Enter also asks.
        </span>
      </div>
    </div>
  );
}
