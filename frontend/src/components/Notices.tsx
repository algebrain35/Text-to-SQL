import { ApiError, NetworkError, type AskResponse } from "../api/client";

/** What a non-success `status` means, in the words a user needs (SPEC 3.5). */
export function statusMessage(response: AskResponse): { title: string; body: string } | null {
  const tries = response.metrics?.attempt_count ?? response.attempts?.length ?? 0;
  switch (response.status) {
    case "success":
      return null;
    case "invalid_sql":
      return {
        title: "No valid query was produced",
        body: `The model did not write SQL that passed checking${tries ? ` in ${tries} ${tries === 1 ? "attempt" : "attempts"}` : ""}. Try rewording the question, or open the attempts below to see what went wrong.`,
      };
    case "execution_error":
      return {
        title: "The query failed when it ran",
        body: "The SQL was valid but the database rejected it. The attempts below show the error.",
      };
    case "no_schema":
      return {
        title: "No matching tables",
        body: "Nothing in this database looked related to the question. Name what you want to count or list, for example “customers” or “orders”.",
      };
    case "unsafe_sql":
      return {
        title: "The query was not run",
        body: "The model wrote something other than a single read-only SELECT, so it was blocked. Try asking for the data directly.",
      };
  }
}

export function StatusNotice({ response }: { response: AskResponse }) {
  const message = statusMessage(response);
  if (!message) return null;
  return (
    <div className="notice error" role="alert">
      <strong>{message.title}</strong>
      <p>{message.body}</p>
    </div>
  );
}

/** The request itself failed: bad input (400), a missing table (404), a service down (503), or no API at all. */
export function errorMessage(error: unknown): { title: string; body: string } {
  if (error instanceof NetworkError) {
    return {
      title: "Cannot reach the API",
      body: `Nothing answered at ${error.url}. Start it with “uvicorn schemagraph.api.main:app”, or set VITE_API_URL if it runs elsewhere.`,
    };
  }
  if (error instanceof ApiError) {
    if (error.status === 400) return { title: "The request was not valid", body: error.detail };
    if (error.status === 404) return { title: "Not found", body: error.detail };
    if (error.status === 503) {
      const wait = error.retryAfter ? ` Try again in about ${error.retryAfter} seconds.` : " Try again in a moment.";
      return { title: "A service is unavailable", body: `${error.detail}${wait}` };
    }
    return { title: `The API returned an error (${error.status})`, body: error.detail };
  }
  return { title: "Something went wrong", body: error instanceof Error ? error.message : "Unknown error." };
}

export function ErrorNotice({ error }: { error: unknown }) {
  const message = errorMessage(error);
  return (
    <div className="notice error" role="alert">
      <strong>{message.title}</strong>
      <p>{message.body}</p>
    </div>
  );
}
