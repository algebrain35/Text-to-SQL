import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApiError, NetworkError, type AskResponse, type ExplainJoin, type ExplainTable } from "../api/client";
import { AttemptsView } from "./AttemptsView";
import { derivedWarnings } from "./Answer";
import { JoinPathView } from "./JoinPathView";
import { ModeSelector } from "./ModeSelector";
import { errorMessage, StatusNotice } from "./Notices";
import { csvOf, numericColumns, ResultTable } from "./ResultTable";
import { TablesUsed } from "./TablesUsed";
import { WarningsView } from "./WarningsView";

const tables: ExplainTable[] = [
  { uid: "db:a.orders", display_name: "a.orders", role: "seed", reason: "Has totals." },
  { uid: "db:a.customer", display_name: "a.customer", role: "bridge", reason: "Links them." },
  { uid: "db:p.person", display_name: "p.person", role: "seed", reason: "Has names." },
];
const joins: ExplainJoin[] = [
  { from: "a.orders.cid", to: "a.customer.cid", source: "declared", text: "Each order has one customer." },
  { from: "a.customer.pid", to: "p.person.id", source: "inferred", text: "Each customer is a person." },
];

describe("ResultTable", () => {
  const result = { columns: ["name", "total"], rows: [["Jon", "8248.99"], ["Eve", null]], row_count: 2, truncated: false };

  it("shows nulls as null, counts rows and right-aligns number columns including their headers", () => {
    render(<ResultTable result={result} />);
    expect(screen.getByText("2 rows")).toBeInTheDocument();
    expect(screen.getByText("null")).toHaveClass("null");
    expect(screen.getByRole("columnheader", { name: "total" })).toHaveClass("num");
    expect(screen.getByRole("columnheader", { name: "name" })).not.toHaveClass("num");
    expect(numericColumns(result)).toEqual([false, true]);
  });

  it("says when the result was cut off and when it is empty", () => {
    const { rerender } = render(<ResultTable result={{ ...result, truncated: true }} />);
    expect(screen.getByText(/Showing the first 2 rows/)).toBeInTheDocument();
    rerender(<ResultTable result={{ columns: ["x"], rows: [], row_count: 0, truncated: false }} />);
    expect(screen.getByText("The query ran and returned no rows.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Download CSV" })).not.toBeInTheDocument();
  });

  it("quotes CSV cells that need it", () => {
    expect(csvOf({ columns: ["a", "b"], rows: [['say "hi"', null], ["x,y", "1"]], row_count: 2, truncated: false })).toBe(
      'a,b\r\n"say ""hi""",\r\n"x,y",1',
    );
  });
});

describe("TablesUsed", () => {
  it("labels matched and connecting tables differently, in words and in shape", () => {
    const { container } = render(<TablesUsed tables={tables} />);
    expect(screen.getAllByText("Matches your question")).toHaveLength(2);
    expect(screen.getByText("Connects other tables")).toBeInTheDocument();
    expect(container.querySelectorAll(".marker.seed")).toHaveLength(2);
    expect(container.querySelectorAll(".marker.bridge")).toHaveLength(1);
  });

  it("renders nothing without tables", () => {
    const { container } = render(<TablesUsed tables={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("JoinPathView", () => {
  it("draws a stop per table and a leg per join, and marks the inferred one", () => {
    const plan = {
      joins: [{ from_column: "db:a.customer.pid", to_column: "db:p.person.id", source: "inferred", confidence: 0.88, cardinality: "N:1", traversal: "many_to_one", weight: 2 }],
    } as never;
    const { container } = render(<JoinPathView tables={tables} joins={joins} plan={plan} />);
    expect(container.querySelectorAll(".stop")).toHaveLength(3);
    expect(container.querySelectorAll(".leg")).toHaveLength(2);
    expect(container.querySelector(".leg.declared")).not.toBeNull();
    const inferred = container.querySelector(".leg.inferred") as HTMLElement;
    expect(within(inferred).getByText(/Inferred from column names and data, not declared/)).toHaveTextContent("confidence 0.88");
    expect(screen.getByText("a.orders.cid = a.customer.cid")).toBeInTheDocument();
  });

  it("lists the joins when they do not form one line", () => {
    const branching = [...joins, { from: "a.customer.sid", to: "a.store.id", source: "declared" as const, text: "Store." }];
    const { container } = render(
      <JoinPathView tables={[...tables, { uid: "db:a.store", display_name: "a.store", role: "bridge", reason: "" }]} joins={branching} />,
    );
    expect(container.querySelector(".trail")).toBeNull();
    expect(container.querySelectorAll(".join-list li")).toHaveLength(3);
  });

  it("renders nothing when there are no joins", () => {
    const { container } = render(<JoinPathView tables={tables} joins={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("AttemptsView", () => {
  it("is collapsed by default and counts failures", async () => {
    const { container } = render(<AttemptsView attempts={[{ sql: "SELECT 1", error: "boom" }, { sql: "SELECT 2", error: null }]} />);
    const details = container.querySelector("details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(screen.getByText("Attempts (2, 1 failed)")).toBeInTheDocument();
    await userEvent.click(screen.getByText(/Attempts/));
    expect(details.open).toBe(true);
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("renders nothing when there were no attempts", () => {
    expect(render(<AttemptsView attempts={[]} />).container).toBeEmptyDOMElement();
  });
});

describe("ModeSelector", () => {
  it("offers the three modes of SPEC 2.3 and reports the choice", async () => {
    const onChange = vi.fn();
    render(<ModeSelector value="graph" onChange={onChange} />);
    expect(screen.getAllByRole("radio").map((r) => (r as HTMLInputElement).value)).toEqual(["graph", "retrieval", "full"]);
    await userEvent.click(screen.getByRole("radio", { name: /Full/ }));
    expect(onChange).toHaveBeenCalledWith("full");
  });
});

describe("notices and warnings", () => {
  const response = (over: Partial<AskResponse>): AskResponse => ({ request_id: "r", question: "q", db_id: "d", mode: "graph", status: "success", ...over });

  it("explains every non-success status and says nothing on success", () => {
    const { container, rerender } = render(<StatusNotice response={response({})} />);
    expect(container).toBeEmptyDOMElement();
    for (const [status, title] of [
      ["invalid_sql", "No valid query was produced"],
      ["execution_error", "The query failed when it ran"],
      ["no_schema", "No matching tables"],
      ["unsafe_sql", "The query was not run"],
    ] as const) {
      rerender(<StatusNotice response={response({ status })} />);
      expect(screen.getByRole("alert")).toHaveTextContent(title);
    }
  });

  it("words request failures for the person: 400, 404, 503 with a wait, no API", () => {
    expect(errorMessage(new ApiError(400, "question: too short")).body).toBe("question: too short");
    expect(errorMessage(new ApiError(404, "No table x")).title).toBe("Not found");
    expect(errorMessage(new ApiError(503, "Neo4j is unreachable.", 30)).body).toBe("Neo4j is unreachable. Try again in about 30 seconds.");
    expect(errorMessage(new NetworkError("http://localhost:8000")).body).toContain("uvicorn schemagraph.api.main:app");
  });

  it("works out warnings from the plan and the joins", () => {
    const plan = { status: "partial", unreachable_seeds: ["db:x.y"], dropped_seeds: ["db:z.w"] } as never;
    const out = derivedWarnings(response({ join_plan: plan, explanation: { summary: "s", tables, join_path: joins, warnings: [] } }));
    expect(out).toHaveLength(3);
    expect(out[0]).toContain("x.y");
    expect(out[1]).toContain("z.w");
    expect(out[2]).toContain("does not declare");
    expect(derivedWarnings(response({}))).toEqual([]);
  });

  it("lists warnings and hides the section when there are none", () => {
    const { container, rerender } = render(<WarningsView warnings={[]} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<WarningsView warnings={["First", "Second"]} />);
    expect(screen.getByRole("heading", { name: "Check these" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });
});
