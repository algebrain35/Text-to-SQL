/*Handles communication between the frontend and FastAPI backend
Uses the API types generated in types.ts
Mock Data can be enabled for testing without connection to backend
 */

import type { components } from "./types";
import * as rawMock from "./mock";

type Schemas = components["schemas"];
export type Mode = Schemas["AskRequest"]["mode"];

//Mode is optional in the frontend because FastAPI defaults to "graph"
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

//Converts mock HTTP errors into ApiError exceptions
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

//Get the API URL from the environment settings
const configuredUrl = import.meta.env.VITE_API_URL;

//Use localhost as the default API URL
let apiUrl = "http://localhost:8000";

//Use the configured API URL if one exists
if (configuredUrl !== null && configuredUrl !== undefined) {
  apiUrl = configuredUrl;
}

//Remove a trailing slash from the URL if there is one
if (apiUrl.endsWith("/")) {
  apiUrl = apiUrl.slice(0, -1);
}

//final API URL
export const API_URL: string = apiUrl;

//Handles error responses from FastAPI, including the error status and message
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

//Handles network errors when the frontend cannot connect to FastAPI
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

  //Returns error message if it is a non empty string
  if (typeof detail === "string" && detail) {
    return detail;
  }

  //Handle multiple error messages
  if (Array.isArray(detail) && detail.length > 0) {

    //Extract the error messages
    const messages = detail.map((d) => {
      const message = (d as { msg?: string }).msg;

      if (message === null || message === undefined) {
        return "";
      }
      else {
        return message;
      }
    });

    //Remove empty messages and combine them into one string
    const filteredMessages = messages.filter(Boolean);
    const combinedMessage = filteredMessages.join("; ")

    //Return the combined errors, or the fallback message
    if (combinedMessage) {
      return combinedMessage;
    }
    else {
      return fallback;
    }
  }

  return fallback;
}

//handles API requests and responses from FastAPI
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {

  let response: Response;

  try {

    //Prepare the HTTP request headers
    const headers: Record<string, string> = {
      Accept: "application/json"
    };

    //Add the Content-Type header when a request body exists
    if (init.body) {
      headers["Content-Type"] = "application/json";
    }

    //Send the request to FastAPI
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: headers
    });

  }
  catch (error) {

    //Preserve errors caused by cancelled requests
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }

    //Otherwise report a network connection error
    throw new NetworkError(API_URL);
  }

  let body: unknown = null;

  //Attempt to read the JSON response
  try {
    body = await response.json();
  }

  //Ignore responses that can't be parsed as JSON
  catch {
  }

  //Handle unsuccessful HTTP responses
  if (!response.ok) {

    const retry = Number(response.headers.get("Retry-After"));

    //Determine whether a valid retry delay was provided
    let retryAfter: number | undefined = undefined;

    if (Number.isFinite(retry) && retry > 0) {
      retryAfter = retry;
    }

    //Determine the fallback error message
    let fallback = response.statusText;

    if (!fallback) {
      fallback = `Request failed (${response.status}).`;
    }

    throw new ApiError(
      response.status,
      detailOf(body, fallback),
      retryAfter
    );
  }
  return body as T;
}

//Retrieves the backend health status
export function getHealth(signal?: AbortSignal): Promise<Health> {

  //Use mock data if mock mode is enabled
  if (USE_MOCKS) {
    return mock.getHealth();
  }

  //if not request the health status from FastAPI
  else {
    return request<Health>("/api/health", { signal });
  }
}

//Retrieves the list of available databases
export function getDatabases(signal?: AbortSignal): Promise<DatabaseInfo[]> {

  //Use mock data if mock enabled
  if (USE_MOCKS) {
    return mock.getDatabases();
  }

  //if not get the databases from FastAPI
  else {
    return request<DatabaseInfo[]>("/api/databases", { signal });
  }
}

//Sends a question to generate SQL and retrieve results
export function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {

  //USe mock data if mock enabled
  if (USE_MOCKS) {
    return mock.ask(req, signal);
  }

  //if not send question to FastAPI
  else {
    return request<AskResponse>("/api/ask", { method: "POST", body: JSON.stringify(req), signal });
  }
}

//Retrieves and plans database joins without calling the LLM
export function plan(req: AskRequest, signal?: AbortSignal): Promise<PlanResponse> {

  //Use mock data if mock enabled
  if (USE_MOCKS) {
    return mock.plan(req, signal);
  }

  //if not get the join plan from FastAPI
  else {
    return request<PlanResponse>("/api/plan", { method: "POST", body: JSON.stringify(req), signal });
  }
}

//Retrieves the list of tables for the selected database
export function listTables(dbId: string, signal?: AbortSignal): Promise<TableSummary[]> {

  //Use mock data if mock enabled
  if (USE_MOCKS) {
    return mock.listTables();
  }

  //if not get the tables from FastAPI
  else {
    return request<TableSummary[]>(`/api/schema/${encodeURIComponent(dbId)}/tables`, { signal });
  }
}

//Retrieves a database table and its relationships
export function getTable(dbId: string, uid: string, signal?: AbortSignal): Promise<TableWithRelationships> {

  //Use mock data if mock enabled
  if (USE_MOCKS) {
    return mock.getTable(uid);
  }

  //if not get the table and relationships from FastAPI
  else {
    return request<TableWithRelationships>(
      `/api/schema/${encodeURIComponent(dbId)}/tables/${encodeURIComponent(uid)}`,
      { signal },
    );
  }

}
