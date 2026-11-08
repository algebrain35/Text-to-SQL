import type { ExplainJoin, ExplainTable } from "../api/client";
import { buildTrail, tableOfColumn } from "./trail";

const table = (name: string, role: ExplainTable["role"] = "seed"): ExplainTable => ({
  uid: `db:${name}`,
  display_name: name,
  role,
  reason: "",
});
const join = (from: string, to: string, source: ExplainJoin["source"] = "declared"): ExplainJoin => ({
  from,
  to,
  source,
  text: `${from} joins ${to}`,
});

const ORDERS = table("sales.orders");
const CUSTOMER = table("sales.customer", "bridge");
const PERSON = table("person.person");

describe("tableOfColumn", () => {
  it("picks the longest table name the column starts with, so dotted names are not cut apart", () => {
    const tables = [table("a.b"), table("a.b.c")];
    expect(tableOfColumn("a.b.c.id", tables)?.display_name).toBe("a.b.c");
    expect(tableOfColumn("a.b.id", tables)?.display_name).toBe("a.b");
    expect(tableOfColumn("x.y.id", tables)).toBeUndefined();
  });
});

describe("buildTrail", () => {
  it("returns null when there are no joins", () => {
    expect(buildTrail([ORDERS], [])).toBeNull();
  });

  it("orders a line of tables and keeps each join between its two tables", () => {
    const trail = buildTrail(
      [ORDERS, CUSTOMER, PERSON],
      [join("sales.orders.customerid", "sales.customer.customerid"), join("sales.customer.personid", "person.person.id")],
    );
    expect(trail?.kind).toBe("chain");
    if (trail?.kind !== "chain") return;
    expect(trail.stops.map((t) => t.display_name)).toEqual(["sales.orders", "sales.customer", "person.person"]);
    expect(trail.legs.map((l) => l.from)).toEqual(["sales.orders.customerid", "sales.customer.personid"]);
  });

  it("finds the line whatever order the joins come in, starting from the table listed first", () => {
    const trail = buildTrail(
      [ORDERS, CUSTOMER, PERSON],
      [join("sales.customer.personid", "person.person.id"), join("sales.orders.customerid", "sales.customer.customerid")],
    );
    expect(trail?.kind === "chain" && trail.stops.map((t) => t.display_name)).toEqual([
      "sales.orders",
      "sales.customer",
      "person.person",
    ]);
  });

  it("keeps the join's own direction even when it points against the line", () => {
    const trail = buildTrail([PERSON, CUSTOMER], [join("sales.customer.personid", "person.person.id")]);
    expect(trail?.kind === "chain" && trail.stops.map((t) => t.display_name)).toEqual(["person.person", "sales.customer"]);
  });

  it("falls back to a list when the joins branch", () => {
    const tables = [CUSTOMER, ORDERS, PERSON, table("sales.store")];
    const trail = buildTrail(tables, [
      join("sales.orders.customerid", "sales.customer.id"),
      join("sales.customer.personid", "person.person.id"),
      join("sales.customer.storeid", "sales.store.id"),
    ]);
    expect(trail?.kind).toBe("list");
  });

  it("falls back to a list when two tables are joined twice or a table is unknown", () => {
    const twice = buildTrail([ORDERS, CUSTOMER], [join("sales.orders.a", "sales.customer.a"), join("sales.orders.b", "sales.customer.b")]);
    expect(twice?.kind).toBe("list");
    expect(buildTrail([ORDERS], [join("sales.orders.a", "elsewhere.t.a")])?.kind).toBe("list");
  });
});
