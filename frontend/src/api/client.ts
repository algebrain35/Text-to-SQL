/**
 * The only module that talks to the backend (SPEC 5.22). The shapes come from `types.ts`, which is generated from
 * the API's /openapi.json (`npm run gen:types` with the API running).
 *
 * Set VITE_USE_MOCKS=true to answer from src/mocks instead of the network (see .env.example).
 */
import type { components } from "./types";
import * as rawMock from "./mock";

type Schemas = components["schemas"];
export type Mode = Schemas["AskRequest"]["mode"];
/** `mode` defaults to "graph" on the server (SPEC 3.5); the generator marks it required because it has a default. */
export type AskRequest = Omit<Schemas["AskRequest"], "mode"> & { mode?: Mode };
export type AskResponse = Schemas["AskResponse"];
export type AskOptions = Schemas["AskOptions"];
export type PlanResponse = Schemas["PlanResponse"];
export type JoinPlan = Schemas["JoinPlan"];
export type Health = Schemas["Health"];
export type DatabaseInfo = Schemas["DatabaseInfo"];
export type TableSummary = Schemas["TableSummary"];
export type TableWithRelationships = Schemas["TableWithRelationships"];
export type ExplainTable = Schemas["ExplainTable"];
export type ExplainJoin = Schemas["ExplainJoin"];
export type Explanation = Schemas["Explanation"];
export type RelationshipMeta = Schemas["RelationshipMeta"];
export type Status = AskResponse["status"];

export const MODES: readonly Mode[] = ["graph", "retrieval", "full"];
/** The mock, with its HTTP-style failures turned into the ApiError the real client throws. */
const mock = {
  getHealth: rawMock.getHealth,
  getDatabases: rawMock.getDatabases,
  ask: (req: AskRequest, signal?: AbortSignal) => asApiError(rawMock.ask(req, signal)),
  plan: (req: AskRequest, signal?: AbortSignal) => asApiError(rawMock.plan(req, signal)),
  listTables: rawMock.listTables,
  getTable: (uid: string) => asApiError(rawMock.getTable(uid)),
};

async function asApiError<T>(call: Promise<T>): Promise<T> {
  try {
    return await call;
  } catch (error) {
    if (error instanceof rawMock.MockHttpError) throw new ApiError(error.status, error.detail);
    throw error;
  }
}

export const USE_MOCKS = import.meta.env.VITE_USE_MOCKS === "true";
export const API_URL: string = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** The API answered with an error status. `detail` is the API's own message (SPEC 3.5: 400, 404 and 503). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly retryAfter?: number;

  constructor(status: number, detail: string, retryAfter?: number) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

/** The request never got an answer: the API is not running, or the address is wrong. */
export class NetworkError extends Error {
  readonly url: string;

  constructor(url: string) {
    super(`Cannot reach the SchemaGraph API at ${url}.`);
    this.name = "NetworkError";
    this.url = url;
  }
}

function detailOf(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((d) => (d as { msg?: string }).msg ?? "").filter(Boolean).join("; ") || fallback;
  }
  return fallback;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new NetworkError(API_URL);
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    /* an error page or an empty body */
  }
  if (!response.ok) {
    const retry = Number(response.headers.get("Retry-After"));
    throw new ApiError(
      response.status,
      detailOf(body, response.statusText || `Request failed (${response.status}).`),
      Number.isFinite(retry) && retry > 0 ? retry : undefined,
    );
  }
  return body as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return USE_MOCKS ? mock.getHealth() : request<Health>("/api/health", { signal });
}

export function getDatabases(signal?: AbortSignal): Promise<DatabaseInfo[]> {
  return USE_MOCKS ? mock.getDatabases() : request<DatabaseInfo[]>("/api/databases", { signal });
}

export function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  return USE_MOCKS
    ? mock.ask(req, signal)
    : request<AskResponse>("/api/ask", { method: "POST", body: JSON.stringify(req), signal });
}

/** Retrieval and join planning only; no model call. */
export function plan(req: AskRequest, signal?: AbortSignal): Promise<PlanResponse> {
  return USE_MOCKS
    ? mock.plan(req, signal)
    : request<PlanResponse>("/api/plan", { method: "POST", body: JSON.stringify(req), signal });
}

export function listTables(dbId: string, signal?: AbortSignal): Promise<TableSummary[]> {
  return USE_MOCKS
    ? mock.listTables()
    : request<TableSummary[]>(`/api/schema/${encodeURIComponent(dbId)}/tables`, { signal });
}

export function getTable(dbId: string, uid: string, signal?: AbortSignal): Promise<TableWithRelationships> {
  return USE_MOCKS
    ? mock.getTable(uid)
    : request<TableWithRelationships>(
        `/api/schema/${encodeURIComponent(dbId)}/tables/${encodeURIComponent(uid)}`,
        { signal },
      );
}
