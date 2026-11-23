const BREAK_BEFORE =
  /^(?:(?:(?:LEFT|RIGHT|FULL|INNER|CROSS)(?:\s+OUTER)?\s+)?JOIN|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION(?:\s+ALL)?|INTERSECT|EXCEPT)\b/i;
const JOIN_START = /^(?:(?:LEFT|RIGHT|FULL|INNER|CROSS)(?:\s+OUTER)?\s+)?JOIN\b/i;

/**
 * Lay out SQL for reading: a new line before each top-level clause, joins indented. Text inside quotes and
 * inside parentheses (subqueries, EXTRACT(year FROM d), window clauses) is left exactly as written. Display
 * only; the original string is what runs and what is copied.
 */
export function formatSql(sql: string): string {
  const text = sql.trim().replace(/;\s*$/, "");
  let out = "";
  let depth = 0;
  let i = 0;
  while (i < text.length) {
    const ch = text[i] as string;
    if (ch === "'" || ch === '"') {
      let j = i + 1;
      while (j < text.length) {
        if (text[j] === ch) {
          if (text[j + 1] === ch) j += 2; // doubled quote is an escaped quote
          else break;
        } else j += 1;
      }
      out += text.slice(i, j + 1);
      i = j + 1;
      continue;
    }
    if (ch === "(") depth += 1;
    if (ch === ")") depth = Math.max(0, depth - 1);
    const atWordStart = i === 0 || !/[\w$]/.test(text[i - 1] as string);
    if (depth === 0 && atWordStart && /[A-Za-z]/.test(ch)) {
      const rest = text.slice(i);
      const match = BREAK_BEFORE.exec(rest);
      if (match && out.trim() !== "") {
        out = out.replace(/\s+$/, "") + (JOIN_START.test(rest) ? "\n  " : "\n");
        out += match[0].replace(/\s+/g, " ");
        i += match[0].length;
        continue;
      }
    }
    out += ch;
    i += 1;
  }
  return out;
}
