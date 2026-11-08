"""Everything embedding-related that retrieval and enrichment share.

- the `Embedder` interface (the same two methods as LangChain's `Embeddings`, so
  a LangChain embeddings object can be passed anywhere an Embedder is wanted)
- two HTTP embedders: `OpenAICompatibleEmbedder` (OpenAI and look-alike servers) and
  `VoyageEmbedder` (Voyage AI, which Anthropic recommends because it has no embedding
  model of its own), chosen by `embedding.provider` in config.yml
- the names of the Neo4j indexes and property that hold one model's vectors
- the text that gets embedded for a table and for a column (pure functions)
- creating and dropping the two Neo4j vector indexes

Retrieval finds tables two ways: through the table's own text (name and
description) and through its columns, so a question about "total due" can find
`salesorderheader` through its `totaldue` column. Both live in the same Neo4j
database, kept apart by the db_id property on every node.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

from schemagraph.config import EmbeddingConfig, get_config
from schemagraph.graph.cypher import use_session
from schemagraph.models import ColumnMeta, TableMeta

log = logging.getLogger(__name__)

DEFAULT_NAME = "default"
TABLE_INDEX = "table_embedding"      # names for the default set of vectors; see embedding_names()
COLUMN_INDEX = "column_embedding"
EMBEDDING_PROP = "embedding"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")

MAX_SAMPLE_VALUES = 3
MAX_SAMPLE_CHARS = 40


class EmbeddingError(RuntimeError):
    """The embedding service failed or returned something unusable."""


class RateLimited(EmbeddingError):
    """The service kept answering HTTP 429 until the wait budget (`max_wait_s`) ran out."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


RATE_LIMIT_BACKOFF = (5.0, 15.0, 30.0, 60.0)  # seconds; the last value repeats
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def retry_after_seconds(header: str | None, body: str = "") -> float | None:
    """How long the service asked us to wait: the Retry-After header (seconds), else the
    `retryDelay` Google puts in the body of its 429 answers. None if neither is there."""
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass  # an HTTP date: not worth parsing, the backoff schedule covers it
    match = _RETRY_DELAY_RE.search(body)
    return float(match.group(1)) if match else None


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingNames(NamedTuple):
    table_index: str
    column_index: str
    prop: str


def embedding_names(name: str = DEFAULT_NAME) -> EmbeddingNames:
    """Index and property names for one set of vectors.

    Vectors from different models cannot be compared, and an index holds one dimension,
    so each model gets its own set. The default name keeps the plain names
    (`table_embedding`, `column_embedding`, `embedding`); any other name is added as a
    suffix (`table_embedding_voyage`). The names go into Cypher text, so they are checked.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"embedding name must be lowercase letters, digits and _, got {name!r}")
    if name == DEFAULT_NAME:
        return EmbeddingNames(TABLE_INDEX, COLUMN_INDEX, EMBEDDING_PROP)
    return EmbeddingNames(f"{TABLE_INDEX}_{name}", f"{COLUMN_INDEX}_{name}", f"{EMBEDDING_PROP}_{name}")


def names_for(embedder: object, name: str | None = None) -> EmbeddingNames:
    """The names to use with `embedder`: an explicit `name`, else the embedder's own, else the default.

    HTTP embedders carry the `name` from config, so vectors are always written and searched
    under the name of the model that made them. Test doubles without a name use the default.
    """
    return embedding_names(name or getattr(embedder, "name", DEFAULT_NAME))


# ---------------------------------------------------------------------------
# Text that gets embedded (pure)
# ---------------------------------------------------------------------------


def humanize(name: str) -> str:
    """'order_detail' -> 'order detail'. Names without underscores are left alone."""
    return name.replace("_", " ")


def _description(description: str | None, source: str | None, include_generated: bool) -> str | None:
    if not description or not description.strip():
        return None
    if source == "generated" and not include_generated:
        return None  # the ablation: embed only what the catalog itself says
    return " ".join(description.split())


MAX_RELATED = 12  # neighbours listed per direction; a hub table would otherwise bury its own description
RELATED_MIN_CONFIDENCE = 0.8  # declared keys are 1.0; weak inferred links would add noise


def table_text(
    table: TableMeta,
    include_generated: bool = True,
    references: list[str] | None = None,
    referenced_by: list[str] | None = None,
) -> str:
    """'sales.salesorderheader: General sales order information.'

    With `references` / `referenced_by` (display names of the tables this one has foreign keys to, and of those
    with keys to it) the text continues 'References: ... Referenced by: ...', so a table that links other tables
    can be found from questions about the things it links.
    """
    text = f"{table.schema_}.{table.name}"
    if humanize(table.name) != table.name:
        text += f" ({humanize(table.name)})"
    desc = _description(table.description, table.description_source, include_generated)
    text = f"{text}: {desc}" if desc else text
    for label, names in (("References", references), ("Referenced by", referenced_by)):
        if names:
            shown = sorted(names)[:MAX_RELATED]
            more = f", and {len(names) - MAX_RELATED} more" if len(names) > MAX_RELATED else ""
            text += f" {label}: {', '.join(shown)}{more}."
    return text


def column_text(table: TableMeta, column: ColumnMeta, include_generated: bool = True) -> str:
    """'sales.salesorderheader.totaldue (numeric): Total due from the customer. Examples: 8248.99, 6383.66'"""
    text = f"{table.schema_}.{table.name}.{column.name}"
    if humanize(column.name) != column.name:
        text += f" ({humanize(column.name)})"
    text += f" ({column.data_type})"
    desc = _description(column.description, column.description_source, include_generated)
    if desc:
        text += f": {desc}"
    samples = [v[:MAX_SAMPLE_CHARS] for v in column.sample_values[:MAX_SAMPLE_VALUES] if v]
    if samples:
        text += f". Examples: {', '.join(samples)}"
    return text


# ---------------------------------------------------------------------------
# Embedding over HTTP
# ---------------------------------------------------------------------------


class _HttpEmbedder:
    """Batching, retries and checks shared by the HTTP embedders.

    A provider supplies the request body (`_payload`) and, if it differs from the OpenAI
    style, where to post (`_url`), how to authenticate (`_auth_headers`) and how to read the
    answer (`_read_vectors`). OpenAI and Voyage answer with {"data": [{"index", "embedding"}, ...]}.
    """

    RETRY_STATUS = {429, 500, 502, 503, 504}
    PATH = "/embeddings"

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
        retries: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.name = config.name
        self._sleep = sleep
        self._retries = retries
        self._clock = clock
        self._next_request_at: float | None = None

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        out: list[list[float]] = []
        size = self.config.batch_size
        for i in range(0, len(texts), size):
            out.extend(self._embed_batch(texts[i : i + size], kind))
        return out

    def _payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        raise NotImplementedError

    def _url(self) -> str:
        return self.config.base_url.rstrip("/") + self.PATH

    def _auth_headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def _read_vectors(self, body: Any) -> list[list[float]]:
        data = sorted(body["data"], key=lambda item: item["index"])
        return [[float(x) for x in item["embedding"]] for item in data]

    def _embed_batch(self, texts: list[str], kind: str) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        key = self.config.api_key
        if key is not None:
            headers.update(self._auth_headers(key.get_secret_value()))
        request = urllib.request.Request(
            self._url(),
            data=json.dumps(self._payload(texts, kind)).encode(),
            headers=headers,
            method="POST",
        )
        body = self._send(request)
        try:
            vectors = self._read_vectors(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"unexpected response from the embedding service: {str(body)[:200]}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        for v in vectors:
            if len(v) != self.config.dimensions:
                raise EmbeddingError(
                    f"{self.config.model} returned {len(v)}-dimensional vectors but config says dimensions: "
                    f"{self.config.dimensions}. Fix the number, or set send_dimensions: true if the model "
                    f"can return shorter vectors."
                )
        return vectors

    def _pace(self) -> None:
        """Space requests out when `requests_per_minute` is set."""
        rpm = self.config.requests_per_minute
        if not rpm:
            return
        now = self._clock()
        start = now if self._next_request_at is None else max(now, self._next_request_at)
        if start > now:
            self._sleep(start - now)
        self._next_request_at = start + 60.0 / rpm

    def _send(self, request: urllib.request.Request) -> Any:
        """POST with retries. Rate limits (429) wait as long as the service asks, at least the
        backoff schedule, until `max_wait_s` is spent; other transient errors retry a few times."""
        waited = 0.0
        rate_limited = 0
        attempt = 0
        while True:
            self._pace()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                raw = exc.read()[:2000].decode(errors="replace")
                detail = raw[:200]
                if exc.code == 429:
                    hinted = retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None, raw)
                    wait = max(hinted or 0.0, RATE_LIMIT_BACKOFF[min(rate_limited, len(RATE_LIMIT_BACKOFF) - 1)])
                    if waited + wait > self.config.max_wait_s:
                        raise RateLimited(
                            f"rate limited (HTTP 429) and the {self.config.max_wait_s:g}s wait budget (max_wait_s) "
                            f"is used up after {waited:g}s: {detail}",
                            hinted,
                        ) from exc
                    log.warning("rate limited by the embedding service; waiting %gs", wait)
                    self._sleep(wait)
                    waited += wait
                    rate_limited += 1
                    continue
                if exc.code in self.RETRY_STATUS and attempt < self._retries:
                    self._sleep(2**attempt)
                    attempt += 1
                    continue
                raise EmbeddingError(f"embedding service returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self._retries:
                    self._sleep(2**attempt)
                    attempt += 1
                    continue
                raise EmbeddingError(f"cannot reach the embedding service at {self.config.base_url}: {exc}") from exc


class OpenAICompatibleEmbedder(_HttpEmbedder):
    """POST {base_url}/embeddings in OpenAI's format. Works with OpenAI and servers that copy its API."""

    def _payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.config.model, "input": texts}
        if self.config.send_dimensions:
            payload["dimensions"] = self.config.dimensions
        return payload  # OpenAI embeds queries and documents the same way


class VoyageEmbedder(_HttpEmbedder):
    """POST {base_url}/embeddings in Voyage AI's format.

    Voyage embeds a search query and a stored document differently, so every request says
    which it is (`input_type`). Shortened vectors are requested with `output_dimension`.
    """

    def _payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.config.model, "input": texts, "input_type": kind}
        if self.config.send_dimensions:
            payload["output_dimension"] = self.config.dimensions
        return payload


class GeminiEmbedder(_HttpEmbedder):
    """Google's Gemini API: POST {base_url}/models/{model}:batchEmbedContents.

    Two generations differ in how a query is told apart from a document:
    `gemini-embedding-001` takes a `taskType` (RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT);
    `gemini-embedding-2` rejects `taskType` and expects a text prefix instead
    ("task: search result | query: ..." and "title: none | text: ..."). Any model name that does
    not start with `gemini-embedding-2` is treated like 001. Shortened vectors are requested with
    `outputDimensionality`; without it the model returns 3072. 001 does not renormalise shortened
    vectors, which is harmless here because the Neo4j indexes use cosine similarity.
    The key goes in the `x-goog-api-key` header, never in the URL.
    """

    def _prefix_mode(self) -> bool:
        return self.config.model.startswith("gemini-embedding-2")

    def _url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/models/{self.config.model}:batchEmbedContents"

    def _auth_headers(self, key: str) -> dict[str, str]:
        return {"x-goog-api-key": key}

    def _payload(self, texts: list[str], kind: str) -> dict[str, Any]:
        requests = []
        for text in texts:
            if self._prefix_mode():
                text = f"task: search result | query: {text}" if kind == "query" else f"title: none | text: {text}"
            request: dict[str, Any] = {"model": f"models/{self.config.model}", "content": {"parts": [{"text": text}]}}
            if not self._prefix_mode():
                request["taskType"] = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
            if self.config.send_dimensions:
                request["outputDimensionality"] = self.config.dimensions
            requests.append(request)
        return {"requests": requests}

    def _read_vectors(self, body: Any) -> list[list[float]]:
        return [[float(x) for x in item["values"]] for item in body["embeddings"]]


EMBEDDERS: dict[str, type[_HttpEmbedder]] = {
    "openai": OpenAICompatibleEmbedder,
    "voyage": VoyageEmbedder,
    "gemini": GeminiEmbedder,
}


def make_embedder(config: EmbeddingConfig, **kwargs: Any) -> Embedder:
    """The embedder for a config's provider."""
    return EMBEDDERS[config.provider](config, **kwargs)


def get_embedder() -> Embedder:
    """The embedder named by the `embedding:` section of config.yml."""
    cfg = get_config().embedding
    if cfg is None:
        raise RuntimeError(
            "config.yml has no 'embedding' section (provider, model, dimensions, ...); see config.example.yml"
        )
    return make_embedder(cfg)


# ---------------------------------------------------------------------------
# Neo4j vector indexes
# ---------------------------------------------------------------------------


def index_statements(dimensions: int, name: str = DEFAULT_NAME) -> list[str]:
    """Cypher to create both indexes. The dimension is checked and written into the text
    because index options cannot always be passed as parameters."""
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or not 1 <= dimensions <= 4096:
        raise ValueError(f"dimensions must be an integer from 1 to 4096, got {dimensions!r}")
    names = embedding_names(name)
    options = f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
    return [
        f"CREATE VECTOR INDEX {names.table_index} IF NOT EXISTS FOR (n:Table) ON (n.{names.prop}) {options}",
        f"CREATE VECTOR INDEX {names.column_index} IF NOT EXISTS FOR (n:Column) ON (n.{names.prop}) {options}",
    ]


def create_vector_indexes(dimensions: int, session: Any | None = None, name: str = DEFAULT_NAME) -> None:
    """Create the table and column vector indexes (cosine) for one set of vectors. Safe to repeat.

    An existing index keeps its original dimension: if you change a model's dimensions,
    call `drop_vector_indexes` for that name first, then re-run `embed_nodes` for every database.
    Dimensions above 2048 need Neo4j 5.18 or later.
    """
    statements = index_statements(dimensions, name)
    with use_session(session) as s:
        for statement in statements:
            s.run(statement).consume()


def drop_vector_indexes(session: Any | None = None, name: str = DEFAULT_NAME) -> None:
    names = embedding_names(name)
    with use_session(session) as s:
        for index in (names.table_index, names.column_index):
            s.run(f"DROP INDEX {index} IF EXISTS").consume()
