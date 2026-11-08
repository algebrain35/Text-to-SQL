import type { Explanation } from "../api/client";

export function ExplanationPanel({ explanation }: { explanation: Explanation }) {
  return (
    <div className="explanation">
      <p className="summary">{explanation.summary}</p>
      {explanation.interpretation && (
        <p className="interpretation">
          <strong>How the question was read:</strong> {explanation.interpretation}
        </p>
      )}
    </div>
  );
}
