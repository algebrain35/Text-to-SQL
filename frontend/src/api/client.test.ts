import { afterEach, beforeEach, vi } from "vitest";

type Client = typeof import("./client");

function reply(status: number, body: unknown, headers: Record<string, string> = {}) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers }));
}

async function load(env: Record<string, string> = {}): Promise<Client> {
  vi.resetModules();
  for (const [key, value] of Object.entries(env)) vi.stubEnv(key, value);
  return import("./client");
}

let fetchMock: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("client against the API", () => {
  it("posts the question as JSON to /api/ask", async () => {
    const client = await load();
    fetchMock.mockReturnValue(reply(200, { status: "success" }));
    const request = { question: "How many?", db_id: "adventureworks", mode: "graph" as const, options: { top_k: 5 } };
    await client.ask(request);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8000/api/ask");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(request);
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("uses VITE_API_URL without a trailing slash", async () => {
    const client = await load({ VITE_API_URL: "http://api.test:9000/" });
    fetchMock.mockReturnValue(reply(200, []));
    await client.getDatabases();
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://api.test:9000/api/databases");
  });

  it("calls the other endpoints on the paths of SPEC section 4", async () => {
    const client = await load();
    fetchMock.mockImplementation(() => reply(200, {}));
    await client.getHealth();
    await client.plan({ question: "q", db_id: "d" });
    await client.listTables("bird_x");
    await client.getTable("adventureworks", "adventureworks:sales.customer");
    expect(fetchMock.mock.calls.map((c) => c[0])).toEqual([
      "http://localhost:8000/api/health",
      "http://localhost:8000/api/plan",
      "http://localhost:8000/api/schema/bird_x/tables",
      "http://localhost:8000/api/schema/adventureworks/tables/adventureworks%3Asales.customer",
    ]);
  });

  it("turns 400, 404 and 503 into an ApiError carrying the API's own message", async () => {
    const client = await load();
    fetchMock.mockReturnValueOnce(reply(400, { detail: "question: String should have at least 1 character" }));
    await expect(client.ask({ question: "", db_id: "d" })).rejects.toMatchObject({ status: 400, detail: "question: String should have at least 1 character" });

    fetchMock.mockReturnValueOnce(reply(404, { detail: "No table x" }));
    await expect(client.getTable("d", "d:x")).rejects.toMatchObject({ status: 404, detail: "No table x" });

    fetchMock.mockReturnValueOnce(reply(503, { detail: "Neo4j is unreachable." }, { "Retry-After": "30" }));
    const error = await client.ask({ question: "q", db_id: "d" }).catch((e) => e);
    expect(error).toBeInstanceOf(client.ApiError);
    expect(error).toMatchObject({ status: 503, detail: "Neo4j is unreachable.", retryAfter: 30 });
  });

  it("keeps a failure the pipeline reports in `status` as an ordinary answer", async () => {
    const client = await load();
    fetchMock.mockReturnValue(reply(200, { status: "invalid_sql", sql: null }));
    await expect(client.ask({ question: "q", db_id: "d" })).resolves.toMatchObject({ status: "invalid_sql" });
  });

  it("says the API is unreachable when the request never gets an answer, and reads non-JSON error pages", async () => {
    const client = await load();
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(client.getHealth()).rejects.toBeInstanceOf(client.NetworkError);

    fetchMock.mockReturnValueOnce(Promise.resolve(new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" })));
    await expect(client.getHealth()).rejects.toMatchObject({ status: 502, detail: "Bad Gateway" });
  });

  it("lets an aborted request surface as an AbortError, not a network error", async () => {
    const client = await load();
    fetchMock.mockRejectedValueOnce(new DOMException("Aborted", "AbortError"));
    await expect(client.getHealth()).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("client with VITE_USE_MOCKS=true", () => {
  it("answers from the samples without touching the network", async () => {
    const client = await load({ VITE_USE_MOCKS: "true" });
    const mock = await import("./mock");
    mock.setMockLatency(0);
    const answer = await client.ask({ question: "my own question", db_id: "adventureworks", mode: "retrieval" });
    expect(answer).toMatchObject({ question: "my own question", mode: "retrieval", status: "success" });
    expect(await client.getDatabases()).toEqual([{ db_id: "adventureworks", table_count: 7, schema_count: expect.any(Number) }]);
    expect((await client.listTables("adventureworks")).length).toBe(7);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows the failures a question starting with mock: asks for", async () => {
    const client = await load({ VITE_USE_MOCKS: "true" });
    (await import("./mock")).setMockLatency(0);
    await expect(client.ask({ question: "mock:unavailable", db_id: "d" })).rejects.toMatchObject({ status: 503 });
    await expect(client.ask({ question: "mock:bad", db_id: "d" })).rejects.toBeInstanceOf(client.ApiError);
    expect(await client.ask({ question: "mock:no_schema", db_id: "d" })).toMatchObject({ status: "no_schema", sql: null });
    expect(await client.ask({ question: "mock:invalid_sql", db_id: "d" })).toMatchObject({ status: "invalid_sql" });
    await expect(client.getTable("d", "adventureworks:nope.table")).rejects.toMatchObject({ status: 404 });
  });
});
