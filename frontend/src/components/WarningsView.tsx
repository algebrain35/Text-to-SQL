export function WarningsView({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <section aria-labelledby="warnings-heading" className="warnings">
      <h3 id="warnings-heading">{warnings.length === 1 ? "Check this" : "Check these"}</h3>
      <ul>
        {warnings.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </section>
  );
}
