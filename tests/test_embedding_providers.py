"""Embedding providers (OpenAI, Voyage/Anthropic), named vector sets, and the config that selects them.

No network and no Neo4j: HTTP goes to a local server, Neo4j to FakeVectorStore.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from schemagraph import config as cfg
from schemagraph.config import EmbeddingConfig
from schemagraph.graph import enrich, retrieve, vector
from tests.test_retrieve_unit import DB, DIM, FakeVectorStore, WordEmbedder, _details, meta  # noqa: F401


# ----------------------------------------------------------------- config


def make_cfg(**kw):
    kw.setdefault("model", "m")
    kw.setdefault("dimensions", 8)
    return EmbeddingConfig(**kw)


def test_openai_is_the_default_provider():
    c = make_cfg()
    assert (c.provider, c.name) == ("openai", "default")
    assert c.base_url == "https://api.openai.com/v1" and c.api_key_env == "OPENAI_API_KEY"


def test_voyage_defaults():
    c = make_cfg(provider="voyage")
    assert c.provider == "voyage" and c.api_key_env == "VOYAGE_API_KEY"
    assert c.base_url == "https://ai.mongodb.com/v1"


def test_anthropic_means_voyage():
    c = make_cfg(provider="anthropic")
    assert c.provider == "voyage" and c.api_key_env == "VOYAGE_API_KEY"


def test_explicit_settings_beat_provider_defaults():
    c = make_cfg(provider="voyage", base_url="https://api.voyageai.com/v1", api_key_env="MY_KEY")
    assert (c.base_url, c.api_key_env) == ("https://api.voyageai.com/v1", "MY_KEY")


def test_explicit_null_key_means_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = make_cfg(base_url="http://localhost:11434/v1", api_key_env=None)
    assert c.api_key_env is None and c.api_key is None


def test_missing_key_variable_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        make_cfg(provider="voyage").api_key


@pytest.mark.parametrize("bad", ["cohere", "", "OpenAI"])
def test_unknown_provider_is_rejected(bad):
    with pytest.raises(ValidationError):
        make_cfg(provider=bad)


@pytest.mark.parametrize("bad", ["Upper", "1abc", "", "a b", "x; DROP INDEX y", "a" * 40, "-x"])
def test_bad_profile_names_are_rejected(bad):
    with pytest.raises(ValidationError):
        make_cfg(name=bad)


def test_config_file_section_loads(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text(
        "postgres:\n  admin: {user: a, password_env: A}\n  readonly: {user: r, password_env: R}\n"
        "embedding:\n  provider: anthropic\n  name: voyage\n  model: voyage-4\n  dimensions: 1024\n"
    )
    c = cfg.load_config(path).embedding
    assert (c.provider, c.name, c.model, c.dimensions) == ("voyage", "voyage", "voyage-4", 1024)


# ----------------------------------------------------------------- names


def test_default_names_are_the_plain_ones():
    assert vector.embedding_names() == ("table_embedding", "column_embedding", "embedding")


def test_other_names_add_a_suffix():
    assert vector.embedding_names("voyage") == ("table_embedding_voyage", "column_embedding_voyage", "embedding_voyage")


@pytest.mark.parametrize("bad", ["", "Upper", "x y", "a;b", "1a", None, "a" * 40])
def test_names_are_checked_before_they_reach_cypher(bad):
    with pytest.raises(ValueError):
        vector.embedding_names(bad)


def test_names_for_prefers_override_then_embedder_then_default():
    class Named:
        name = "voyage"

    assert vector.names_for(Named()).prop == "embedding_voyage"
    assert vector.names_for(Named(), "other").prop == "embedding_other"
    assert vector.names_for(object()).prop == "embedding"


class Recorder:
    def __init__(self):
        self.statements = []

    def run(self, statement, **params):
        self.statements.append(statement)
        return self

    def consume(self):
        return None


def test_index_statements_use_the_name_and_dimension():
    t, c = vector.index_statements(1024, "voyage")
    assert "table_embedding_voyage" in t and "(n.embedding_voyage)" in t and "`vector.dimensions`: 1024" in t
    assert "column_embedding_voyage" in c and "FOR (n:Column)" in c
    assert "table_embedding " in vector.index_statements(8)[0]


def test_create_and_drop_touch_only_the_named_indexes():
    rec = Recorder()
    vector.create_vector_indexes(8, rec, name="voyage")
    vector.drop_vector_indexes(rec, name="voyage")
    assert len(rec.statements) == 4
    assert all("voyage" in s for s in rec.statements)
    assert rec.statements[2:] == ["DROP INDEX table_embedding_voyage IF EXISTS", "DROP INDEX column_embedding_voyage IF EXISTS"]


# ----------------------------------------------------------------- HTTP providers


class Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        n = len(body["input"])
        dims = body.get("output_dimension") or body.get("dimensions") or 3
        raw = json.dumps({"data": [{"index": i, "embedding": [0.5] * dims} for i in range(n)]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    Handler.seen = []
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_voyage_marks_queries_and_documents(server, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    emb = vector.make_embedder(make_cfg(provider="voyage", base_url=server, dimensions=3, model="voyage-4"))
    assert isinstance(emb, vector.VoyageEmbedder)
    emb.embed_documents(["a", "b"])
    emb.embed_query("q")
    (p1, auth1, b1), (p2, _, b2) = Handler.seen
    assert p1 == "/v1/embeddings" and auth1 == "Bearer vk"
    assert (b1["model"], b1["input"], b1["input_type"]) == ("voyage-4", ["a", "b"], "document")
    assert (b2["input"], b2["input_type"]) == (["q"], "query")
    assert "dimensions" not in b1 and "output_dimension" not in b1


def test_voyage_shortens_with_output_dimension(server, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    emb = vector.make_embedder(make_cfg(provider="anthropic", base_url=server, dimensions=5, send_dimensions=True))
    assert len(emb.embed_query("q")) == 5
    body = Handler.seen[0][2]
    assert body["output_dimension"] == 5 and "dimensions" not in body


def test_openai_payload_has_no_input_type(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    emb = vector.make_embedder(make_cfg(base_url=server, dimensions=7, send_dimensions=True))
    assert isinstance(emb, vector.OpenAICompatibleEmbedder)
    emb.embed_documents(["a"])
    emb.embed_query("q")
    for _, auth, body in Handler.seen:
        assert auth == "Bearer ok" and "input_type" not in body and body["dimensions"] == 7


def test_dimension_mismatch_says_how_to_fix_it(server, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    emb = vector.make_embedder(make_cfg(provider="voyage", base_url=server, dimensions=9))  # server returns 3
    with pytest.raises(vector.EmbeddingError, match="send_dimensions"):
        emb.embed_query("q")


def test_embedders_carry_their_profile_name(server, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "vk")
    assert vector.make_embedder(make_cfg(provider="voyage", name="v", base_url=server)).name == "v"
    assert vector.make_embedder(make_cfg(api_key_env=None)).name == "default"


def test_get_embedder_follows_the_config(monkeypatch):
    class App:
        embedding = make_cfg(provider="anthropic")

    monkeypatch.setattr(vector, "get_config", lambda: App())
    assert isinstance(vector.get_embedder(), vector.VoyageEmbedder)

    class NoEmbedding:
        embedding = None

    monkeypatch.setattr(vector, "get_config", lambda: NoEmbedding())
    with pytest.raises(RuntimeError, match="embedding"):
        vector.get_embedder()


# ----------------------------------------------------------------- two models in one graph


class SeedEmbedder(WordEmbedder):
    """A word embedder whose hashing differs by seed, standing in for a second model."""

    def __init__(self, name, seed):
        super().__init__()
        self.name, self._seed = name, seed

    def _vec(self, text):
        import zlib

        v = [0.0] * DIM
        for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
            v[zlib.crc32(f"{self._seed}{w}".encode()) % DIM] += 1.0
        return v


def test_two_models_live_side_by_side(meta, monkeypatch):
    store = FakeVectorStore()
    store.add_metadata(meta)
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    a, b = SeedEmbedder("default", "a"), SeedEmbedder("voyage", "b")
    enrich.embed_nodes(DB, a, store)
    enrich.embed_nodes(DB, b, store)

    node = next(iter(store.tables.values()))
    assert "embedding" in node and "embedding_voyage" in node and node["embedding"] != node["embedding_voyage"]

    # each model is searched through its own index, with its own query vector
    store.queries.clear()
    retrieve.retrieve_seed_tables(DB, "customer", 3, embedder=a, session=store)
    retrieve.retrieve_seed_tables(DB, "customer", 3, embedder=b, session=store)
    indexes = [p["index"] for _, p in store.queries if "index" in p]
    assert indexes == ["table_embedding", "column_embedding", "table_embedding_voyage", "column_embedding_voyage"]


def test_searching_a_name_that_was_never_embedded_finds_nothing(meta, monkeypatch):
    store = FakeVectorStore()
    store.add_metadata(meta)
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    enrich.embed_nodes(DB, SeedEmbedder("default", "a"), store)
    assert retrieve.retrieve_seed_tables(DB, "customer", 3, embedder=SeedEmbedder("other", "a"), session=store) == []


def test_explicit_name_overrides_the_embedders(meta, monkeypatch):
    store = FakeVectorStore()
    store.add_metadata(meta)
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    emb = SeedEmbedder("default", "a")
    enrich.embed_nodes(DB, emb, store, name="trial")
    assert "embedding_trial" in next(iter(store.tables.values()))
    assert retrieve.retrieve_seed_tables(DB, "customer", 3, embedder=emb, session=store, name="trial")


# ----------------------------------------------------------------- Gemini


def test_gemini_defaults():
    c = make_cfg(provider="gemini", model="gemini-embedding-2")
    assert c.provider == "gemini" and c.api_key_env == "GEMINI_API_KEY"
    assert c.base_url == "https://generativelanguage.googleapis.com/v1beta"


class GeminiHandler(BaseHTTPRequestHandler):
    seen: list = []
    status_codes: list = []        # popped one per request: a status or (status, headers, body); empty means 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        GeminiHandler.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if GeminiHandler.status_codes:
            step = GeminiHandler.status_codes.pop(0)          # a status, or (status, headers, body text)
            code, extra, text = (step, {}, "") if isinstance(step, int) else step
            if code != 200:
                self.send_response(code)
                for key, value in extra.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(text.encode())))
                self.end_headers()
                self.wfile.write(text.encode())
                return
        dims = body["requests"][0].get("outputDimensionality", 3)
        raw = json.dumps({"embeddings": [{"values": [0.25] * dims} for _ in body["requests"]]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def gemini_server():
    GeminiHandler.seen, GeminiHandler.status_codes = [], []
    srv = HTTPServer(("127.0.0.1", 0), GeminiHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1beta"
    srv.shutdown()


def gemini(server, monkeypatch, **kw):
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    kw.setdefault("model", "gemini-embedding-001")
    kw.setdefault("dimensions", 3)
    return vector.make_embedder(make_cfg(provider="gemini", base_url=server, **kw), sleep=lambda s: None)


def test_gemini_001_uses_task_types_and_the_key_header(gemini_server, monkeypatch):
    emb = gemini(gemini_server, monkeypatch)
    assert isinstance(emb, vector.GeminiEmbedder)
    assert emb.embed_documents(["a", "b"]) == [[0.25] * 3, [0.25] * 3]
    emb.embed_query("q")
    (path, headers, doc_body), (_, _, query_body) = GeminiHandler.seen
    assert path == "/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert headers["x-goog-api-key"] == "gk" and "authorization" not in headers
    assert doc_body["requests"] == [
        {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}, "taskType": "RETRIEVAL_DOCUMENT"}
        for t in ("a", "b")
    ]
    assert query_body["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert "outputDimensionality" not in doc_body["requests"][0]


def test_gemini_2_uses_text_prefixes_and_no_task_type(gemini_server, monkeypatch):
    emb = gemini(gemini_server, monkeypatch, model="gemini-embedding-2")
    emb.embed_documents(["orders table"])
    emb.embed_query("how many orders")
    (path, _, doc_body), (_, _, query_body) = GeminiHandler.seen
    assert path == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
    doc, query = doc_body["requests"][0], query_body["requests"][0]
    assert doc["content"]["parts"][0]["text"] == "title: none | text: orders table"
    assert query["content"]["parts"][0]["text"] == "task: search result | query: how many orders"
    assert "taskType" not in doc and "taskType" not in query


def test_gemini_asks_for_shortened_vectors_only_when_told_to(gemini_server, monkeypatch):
    emb = gemini(gemini_server, monkeypatch, dimensions=5, send_dimensions=True)
    assert len(emb.embed_query("q")) == 5
    assert GeminiHandler.seen[0][2]["requests"][0]["outputDimensionality"] == 5


def test_gemini_batches_by_batch_size_and_keeps_order(gemini_server, monkeypatch):
    emb = gemini(gemini_server, monkeypatch, batch_size=2)
    assert len(emb.embed_documents(list("abcde"))) == 5
    assert [len(body["requests"]) for _, _, body in GeminiHandler.seen] == [2, 2, 1]


def test_gemini_retries_rate_limits_then_succeeds(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [429, 503]
    emb = gemini(gemini_server, monkeypatch)
    assert emb.embed_query("q") == [0.25] * 3
    assert len(GeminiHandler.seen) == 3


def test_gemini_dimension_mismatch_and_bad_shape_are_clear_errors(gemini_server, monkeypatch):
    emb = gemini(gemini_server, monkeypatch, dimensions=9)          # server returns 3 without send_dimensions
    with pytest.raises(vector.EmbeddingError, match="send_dimensions"):
        emb.embed_query("q")

    class Odd(vector.GeminiEmbedder):
        def _send(self, request):
            return {"data": []}

    odd = Odd(make_cfg(provider="gemini", model="gemini-embedding-001", api_key_env=None))
    with pytest.raises(vector.EmbeddingError, match="unexpected response"):
        odd.embed_query("q")


def test_gemini_key_is_not_in_the_url(gemini_server, monkeypatch):
    gemini(gemini_server, monkeypatch).embed_query("q")
    assert "gk" not in GeminiHandler.seen[0][0] and "key=" not in GeminiHandler.seen[0][0]


# ----------------------------------------------------------------- rate limits


def limited(server, monkeypatch, **kw):
    """A Gemini embedder that records its sleeps instead of sleeping."""
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    sleeps = []
    cfg = make_cfg(provider="gemini", base_url=server, model="gemini-embedding-001", dimensions=3, **kw)
    return vector.make_embedder(cfg, sleep=sleeps.append), sleeps


def test_retry_after_header_and_google_retry_delay_are_understood():
    assert vector.retry_after_seconds("40") == 40.0
    assert vector.retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") is None
    body = '{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}}'
    assert vector.retry_after_seconds(None, body) == 34.0
    assert vector.retry_after_seconds("7", body) == 7.0                      # the header wins
    assert vector.retry_after_seconds(None, "") is None


def test_a_rate_limit_waits_as_long_as_the_service_asks(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [(429, {"Retry-After": "40"}, "")]
    emb, sleeps = limited(gemini_server, monkeypatch)
    assert emb.embed_query("q") == [0.25] * 3
    assert sleeps == [40.0]


def test_a_rate_limit_without_a_hint_follows_the_backoff_schedule(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [429, 429, 429]
    emb, sleeps = limited(gemini_server, monkeypatch)
    assert emb.embed_query("q") == [0.25] * 3
    assert sleeps == [5.0, 15.0, 30.0]


def test_the_wait_budget_ends_in_rate_limited(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [429] * 10
    emb, sleeps = limited(gemini_server, monkeypatch, max_wait_s=20)
    with pytest.raises(vector.RateLimited, match="wait budget.*max_wait_s") as caught:
        emb.embed_query("q")
    assert isinstance(caught.value, vector.EmbeddingError)
    assert sleeps == [5.0, 15.0] and len(GeminiHandler.seen) == 3         # 5 + 15 = 20, the next 30 would exceed it


def test_zero_budget_gives_up_at_once(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [429] * 3
    emb, sleeps = limited(gemini_server, monkeypatch, max_wait_s=0)
    with pytest.raises(vector.RateLimited):
        emb.embed_query("q")
    assert sleeps == [] and len(GeminiHandler.seen) == 1


def test_a_long_hint_beyond_the_budget_does_not_wait_at_all(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [(429, {"Retry-After": "3600"}, "")]      # e.g. a daily quota
    emb, sleeps = limited(gemini_server, monkeypatch, max_wait_s=120)
    with pytest.raises(vector.RateLimited) as caught:
        emb.embed_query("q")
    assert sleeps == [] and caught.value.retry_after == 3600.0


def test_each_request_gets_a_fresh_budget(gemini_server, monkeypatch):
    GeminiHandler.status_codes = [429, 200, 429]
    emb, sleeps = limited(gemini_server, monkeypatch, max_wait_s=5, batch_size=1)
    assert len(emb.embed_documents(["a", "b"])) == 2
    assert sleeps == [5.0, 5.0]


def test_requests_per_minute_spaces_requests_out(gemini_server, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    now = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    cfg = make_cfg(provider="gemini", base_url=gemini_server, model="gemini-embedding-001", dimensions=3,
                   requests_per_minute=30, batch_size=1)
    emb = vector.make_embedder(cfg, sleep=sleep, clock=lambda: now[0])
    emb.embed_documents(["a", "b", "c"])
    assert sleeps == [2.0, 2.0]                                            # 30 per minute: one every 2 seconds
    now[0] += 10                                                           # a long pause: no catch-up burst, no wait
    emb.embed_query("d")
    assert sleeps == [2.0, 2.0]


def test_rate_limit_settings_and_gemini_batch_default():
    c = make_cfg(provider="gemini")
    assert (c.batch_size, c.max_wait_s, c.requests_per_minute) == (100, 120.0, None)
    assert make_cfg(provider="gemini", batch_size=10).batch_size == 10
    assert make_cfg().batch_size == 64 and make_cfg(provider="voyage").batch_size == 64
    with pytest.raises(ValidationError):
        make_cfg(requests_per_minute=0)
    with pytest.raises(ValidationError):
        make_cfg(max_wait_s=-1)


# ----------------------------------------------------------------- query cache


def test_query_cache_is_off_unless_configured():
    assert make_cfg().query_cache is None
    assert vector.make_embedder(make_cfg(api_key_env=None))._query_cache is None


def test_a_cached_question_is_not_sent_again(server, tmp_path):
    cache = str(tmp_path / "cache" / "q.sqlite")
    emb = vector.make_embedder(make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=cache))
    first = emb.embed_query("how many orders")
    assert emb.embed_query("how many orders") == first and len(Handler.seen) == 1

    again = vector.make_embedder(make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=cache))
    assert again.embed_query("how many orders") == first and len(Handler.seen) == 1      # a new process shares it
    again.embed_query("a different question")
    assert len(Handler.seen) == 2


def test_the_cache_never_returns_another_models_vector(server, tmp_path):
    cache = str(tmp_path / "q.sqlite")
    for model in ("model-a", "model-b"):
        vector.make_embedder(
            make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=cache, model=model)
        ).embed_query("same question")
    assert len(Handler.seen) == 2


def test_documents_are_never_cached(server, tmp_path):
    emb = vector.make_embedder(make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=str(tmp_path / "q.sqlite")))
    emb.embed_documents(["a"])
    emb.embed_documents(["a"])
    assert len(Handler.seen) == 2


def test_an_unusable_cache_file_is_skipped_not_fatal(server, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    emb = vector.make_embedder(
        make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=str(blocker / "q.sqlite")))
    assert len(emb.embed_query("q")) == 3 and len(emb.embed_query("q")) == 3
    assert len(Handler.seen) == 2


def test_a_cached_vector_of_the_wrong_size_is_replaced(server, tmp_path):
    cache = vector.QueryCache(tmp_path / "q.sqlite")
    config = make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=str(cache.path))
    cache.put(cache.key(config, "q"), [1.0, 2.0])          # left by a config with other dimensions
    assert len(vector.make_embedder(config).embed_query("q")) == 3 and len(Handler.seen) == 1


def test_prefetch_embeds_every_new_question_in_batches_and_caches_them(server, tmp_path):
    cache = str(tmp_path / "q.sqlite")
    config = make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=cache, batch_size=2)
    emb = vector.make_embedder(config)
    questions = ["a", "b", "c", "d", "e", "a"]                     # a repeated question is fetched once
    assert emb.prefetch_queries(questions) == 5
    assert [len(body["input"]) for _, _, body in Handler.seen] == [2, 2, 1]

    assert [len(emb.embed_query(q)) for q in questions] == [3] * 6
    assert emb.prefetch_queries(questions) == 0 and emb.prefetch_queries(questions + ["f"]) == 1
    assert len(Handler.seen) == 4                                   # only "f" went out


def test_prefetch_keeps_finished_batches_when_a_later_one_fails(server, tmp_path):
    config = make_cfg(base_url=server, dimensions=3, api_key_env=None, query_cache=str(tmp_path / "q.sqlite"), batch_size=2)
    emb = vector.make_embedder(config)
    real = emb._embed_batch
    calls = []

    def flaky(texts, kind):
        calls.append(texts)
        if len(calls) == 2:
            raise vector.RateLimited("quota")
        return real(texts, kind)

    emb._embed_batch = flaky
    with pytest.raises(vector.RateLimited):
        emb.prefetch_queries(["a", "b", "c", "d"])
    emb._embed_batch = real
    assert emb.prefetch_queries(["a", "b", "c", "d"]) == 2          # the first batch was kept


def test_prefetch_does_nothing_without_a_cache(server):
    emb = vector.make_embedder(make_cfg(base_url=server, dimensions=3, api_key_env=None))
    assert emb.prefetch_queries(["a", "b"]) == 0 and Handler.seen == []
