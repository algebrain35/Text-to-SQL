import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, vi } from "vitest";

async function openApp() {
  vi.resetModules();
  vi.stubEnv("VITE_USE_MOCKS", "true");
  (await import("./api/mock")).setMockLatency(0);
  const { default: App } = await import("./App");
  render(<App />);
  await screen.findByRole("option", { name: /adventureworks/ });
  return userEvent.setup();
}

async function ask(user: ReturnType<typeof userEvent.setup>, question: string) {
  await user.type(screen.getByLabelText("Your question"), question);
  await user.click(screen.getByRole("button", { name: "Ask" }));
}

beforeEach(() => {
  window.history.replaceState({}, "", "/");
});

describe("App (sample data)", () => {
  it("loads databases and services, and keeps Ask off until there is a question", async () => {
    await openApp();
    expect(screen.getByLabelText("Database")).toHaveValue("adventureworks");
    expect(await screen.findByText("Services ready")).toBeInTheDocument();
    expect(screen.getByText(/Showing sample data/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(screen.getByText(/Ask a question in plain English/)).toBeInTheDocument();
  });

  it("shows the answer with every part of SPEC 5.22: result, SQL, tables, path, explanation, attempts closed", async () => {
    const user = await openApp();
    await ask(user, "total order value per customer");

    expect(await screen.findByRole("heading", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByText(/This adds up the total value of all orders/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "total_order_value" })).toBeInTheDocument();
    expect(screen.getByText("8248.99")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "SQL" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Tables used" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How the tables connect" })).toBeInTheDocument();
    expect(screen.getAllByText("Connects other tables").length).toBeGreaterThan(0);
    expect(document.querySelector("details.attempts")).not.toHaveAttribute("open");
    expect(screen.queryByRole("group", { name: "Schema mode" })).not.toBeInTheDocument();
    expect(screen.queryByText("Developer details")).not.toBeInTheDocument();
  });

  it("asks from the keyboard with Ctrl+Enter", async () => {
    const user = await openApp();
    await user.type(screen.getByLabelText("Your question"), "how many orders{Control>}{Enter}{/Control}");
    expect(await screen.findByRole("heading", { name: "Answer" })).toBeInTheDocument();
  });

  it("shows the mode selector and developer details only in the developer view, and sends the chosen mode", async () => {
    const user = await openApp();
    await user.click(screen.getByLabelText("Developer view"));
    await user.click(screen.getByRole("radio", { name: /Retrieval/ }));
    await ask(user, "how many orders");
    await screen.findByRole("heading", { name: "Answer" });
    expect(screen.getByRole("heading", { name: "Developer details" })).toBeInTheDocument();
    expect(screen.getByText("retrieval", { selector: "dd" })).toBeInTheDocument();   // the mode that was sent

    await user.click(screen.getByLabelText("Developer view"));
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(screen.queryByText("Developer details")).not.toBeInTheDocument();
  });

  it("opens in the developer view with ?dev=1 and can plan without asking the model", async () => {
    window.history.replaceState({}, "", "/?dev=1");
    const user = await openApp();
    expect(screen.getByLabelText("Developer view")).toBeChecked();
    await user.type(screen.getByLabelText("Your question"), "orders by person");
    await user.click(screen.getByRole("button", { name: "Plan only (no model call)" }));
    expect(await screen.findByRole("heading", { name: "Plan (no model call)" })).toBeInTheDocument();
    expect(screen.getByText("Retrieved tables")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "SQL" })).not.toBeInTheDocument();
  });

  it("shows what failed when the pipeline reports a failure, and still lists the attempts", async () => {
    const user = await openApp();
    await ask(user, "mock:invalid_sql");
    expect(await screen.findByRole("heading", { name: "No answer" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("No valid query was produced");
    expect(screen.getByText("Attempts (3, 3 failed)")).toBeInTheDocument();
  });

  it("shows a service outage as a message with what to do, not as an answer", async () => {
    const user = await openApp();
    await ask(user, "mock:unavailable");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("A service is unavailable");
    expect(alert).toHaveTextContent("Neo4j is unreachable");
    expect(screen.queryByRole("heading", { name: "Answer" })).not.toBeInTheDocument();
  });

  it("browses the tables of the database and how they link", async () => {
    const user = await openApp();
    await user.click(screen.getByText(/Browse the tables in adventureworks/));
    const first = await screen.findByRole("button", { name: /sales\.salesorderheader/ });
    await user.click(first);
    expect(await screen.findByRole("heading", { name: "Points to" })).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader", { name: "Column" }).length).toBeGreaterThan(0);
  });

  it("clears the answer when the database changes, so it never describes the wrong one", async () => {
    const user = await openApp();
    await ask(user, "how many orders");
    await screen.findByRole("heading", { name: "Answer" });
    await waitFor(() => expect(screen.getByLabelText("Database")).toBeEnabled());
    await user.selectOptions(screen.getByLabelText("Database"), "adventureworks");
    expect(screen.getByLabelText("Database")).toHaveValue("adventureworks");
  });
});
