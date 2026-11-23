import { formatSql } from "./sql";

describe("formatSql", () => {
  it("starts a line at each top-level clause and indents joins", () => {
    const sql = "SELECT a FROM t JOIN u ON t.id = u.id LEFT JOIN v ON v.id = u.id WHERE a > 1 GROUP BY a ORDER BY a LIMIT 5;";
    expect(formatSql(sql)).toBe(
      ["SELECT a", "FROM t", "  JOIN u ON t.id = u.id", "  LEFT JOIN v ON v.id = u.id", "WHERE a > 1", "GROUP BY a", "ORDER BY a", "LIMIT 5"].join("\n"),
    );
  });

  it("leaves subqueries, function arguments and window clauses on their line", () => {
    const sql = "SELECT EXTRACT(year FROM d), RANK() OVER (PARTITION BY x ORDER BY y) FROM (SELECT d, x, y FROM t WHERE z = 1) s";
    expect(formatSql(sql)).toBe(
      "SELECT EXTRACT(year FROM d), RANK() OVER (PARTITION BY x ORDER BY y)\nFROM (SELECT d, x, y FROM t WHERE z = 1) s",
    );
  });

  it("does not touch text in quotes or names that merely contain a keyword", () => {
    const sql = "SELECT 'a FROM b WHERE c', \"order by\", from_date, left_name FROM t WHERE n = 'it''s FROM here'";
    expect(formatSql(sql)).toBe(
      "SELECT 'a FROM b WHERE c', \"order by\", from_date, left_name\nFROM t\nWHERE n = 'it''s FROM here'",
    );
  });

  it("does not mistake the LEFT() function for LEFT JOIN, and keeps already-formatted SQL stable", () => {
    expect(formatSql("SELECT LEFT(name, 3) FROM t")).toBe("SELECT LEFT(name, 3)\nFROM t");
    const once = formatSql("SELECT a FROM t WHERE b = 1");
    expect(formatSql(once)).toBe(once);
  });

  it("handles empty and unbalanced input without throwing", () => {
    expect(formatSql("")).toBe("");
    expect(() => formatSql("SELECT (a FROM t WHERE 'x")).not.toThrow();
  });
});
