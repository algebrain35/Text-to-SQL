"""vector.py, enrich.py and retrieve.py without Neo4j.

FakeVectorStore plays the part of Neo4j's two vector indexes, which are global:
nodes of every db_id sit in the same index, so top-n comes before the db_id filter.
"""

import json
import math
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from schemagraph.config import EmbeddingConfig
from schemagraph.extraction import extract
from schemagraph.graph import enrich, ingest, queries, retrieve, vector

SAMPLE = Path(__file__).parent.parent / "contracts" / "examples" / "metadata.adventureworks.sample.json"
DB = "adventureworks"
DIM = 256


# --------------------------------------------------------------------------- fakes


class WordEmbedder:
    """Hashes words into a DIM-dimensional bag-of-words vector. Same words -> similar."""

    def __init__(self):
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vec(self, text):
        v = [0.0] * DIM
        for w in "".join(c if c.isalnum() else " " for c in text.lower()).split():
            v[zlib.crc32(w.encode()) % DIM] += 1.0
        return v

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vec(text)

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vec(t) for t in texts]


def cosine01(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return 0.0 if not na or not nb else (1 + sum(x * y for x, y in zip(a, b)) / (na * nb)) / 2


class Rec(dict):
    def data(self):
        return dict(self)


class FakeVectorStore:
    """Holds Table/Column nodes for several databases; answers SET and index queries."""

    def __init__(self):
        self.tables: dict[str, dict] = {}   # uid -> props
        self.columns: dict[str, dict] = {}
        self.col_table: dict[str, str] = {}
        self.queries: list[tuple[str, dict]] = []
        self.writes: list[int] = []         # rows per embedding write, to see that work is saved in chunks

    def add_metadata(self, meta, db_id=None):
        db_id = db_id or meta.db_id
        for t in meta.tables:
            self.tables[t.uid] = {"uid": t.uid, "db_id": db_id, "name": t.name}
        for c in meta.columns:
            self.columns[c.uid] = {"uid": c.uid, "db_id": db_id, "name": c.name}
            self.col_table[c.uid] = c.table_uid

    # neo4j session API
    def execute_write(self, work, rows):
        return work(self, rows)

    def execute_read(self, work):
        return work(self)

    def run(self, query, **params):
        self.queries.append((query, params))
        if query.startswith("UNWIND $rows"):
            label = "Table" if "(n:Table" in query else "Column"
            store = self.tables if label == "Table" else self.columns
            prop = re.search(r"SET n\.(\w+) = row\.vec", query).group(1)
            hash_prop = re.search(r"n\.(\w+) = row\.hash", query)
            for row in params["rows"]:
                store[row["uid"]][prop] = row["vec"]
                if hash_prop:
                    store[row["uid"]][hash_prop.group(1)] = row["hash"]
            self.writes.append(len(params["rows"]))
            return self
        if query.startswith("MATCH (n:") and " AS hash" in query:
            label = re.match(r"MATCH \(n:(\w+)", query).group(1)
            hash_prop = re.search(r"n\.(\w+) AS hash", query).group(1)
            nodes = self.tables if label == "Table" else self.columns
            return [Rec(uid=uid, hash=p.get(hash_prop)) for uid, p in nodes.items() if p["db_id"] == params["db_id"]]
        if "queryNodes" in query:
            if params["index"] == vector.TABLE_INDEX:
                nodes, is_col = self.tables, False
            else:
                nodes, is_col = self.columns, True
            prop = params["index"].split("_", 1)[1]   # table_embedding[_name] -> embedding[_name]
            scored = sorted(
                ((cosine01(p[prop], params["vector"]), uid) for uid, p in nodes.items() if prop in p),
                key=lambda x: (-x[0], x[1]),
            )[: params["n"]]  # top n over ALL databases, then the db_id filter
            out = []
            for score, uid in scored:
                if "node.db_id = $db_id" in query and nodes[uid]["db_id"] != params["db_id"]:
                    continue
                if is_col:
                    out.append(Rec(uid=self.col_table[uid], column=nodes[uid]["name"], score=score))
                else:
                    out.append(Rec(uid=uid, score=score))
            return out
        raise AssertionError(f"unexpected query: {query}")

    def consume(self):
        return None


@pytest.fixture(scope="module")
def meta():
    return extract.load_metadata(SAMPLE)


# --------------------------------------------------------------------------- text builders


def _table(meta, name):
    return next(t for t in meta.tables if t.name == name)


def test_table_text_has_name_and_description(meta):
    t = _table(meta, "salesorderheader").model_copy(update={"description": "General  sales\norder info.", "description_source": "catalog"})
    assert vector.table_text(t) == "sales.salesorderheader: General sales order info."


def test_table_text_humanizes_underscores():
    from schemagraph.models import TableMeta

    t = TableMeta(uid="x:s.order_detail", schema="s", name="order_detail")
    assert vector.table_text(t) == "s.order_detail (order detail)"


def test_generated_descriptions_can_be_left_out(meta):
    t = _table(meta, "customer").model_copy(update={"description": "A buyer.", "description_source": "generated"})
    assert "A buyer." in vector.table_text(t)
    assert vector.table_text(t, include_generated=False) == vector.table_text(
        t.model_copy(update={"description": None, "description_source": None})
    )


def test_column_text_has_type_description_and_trimmed_samples(meta):
    t = _table(meta, "salesorderheader")
    c = next(c for c in meta.columns if c.table_uid == t.uid).model_copy(
        update={"description": "The total.", "description_source": "catalog", "sample_values": ["a" * 100, "b", "c", "d"]}
    )
    text = vector.column_text(t, c)
    assert text.startswith(f"{t.schema_}.{t.name}.{c.name}")
    assert f"({c.data_type})" in text and "The total." in text
    assert "a" * 40 in text and "a" * 41 not in text
    assert text.endswith("Examples: " + ", ".join(["a" * 40, "b", "c"]))


# --------------------------------------------------------------------------- indexes


def test_index_statements_cosine_and_dimension():
    for s in vector.index_statements(1536):
        assert "IF NOT EXISTS" in s and "`vector.dimensions`: 1536" in s and "'cosine'" in s
    assert "table_embedding" in vector.index_statements(8)[0] and "column_embedding" in vector.index_statements(8)[1]


@pytest.mark.parametrize("bad", [0, -1, 5000, "8; DROP", 1.5, True])
def test_index_statements_reject_bad_dimensions(bad):
    with pytest.raises(ValueError):
        vector.index_statements(bad)


# --------------------------------------------------------------------------- rollup


def test_rollup_takes_best_score_per_table():
    seeds = retrieve.rollup(
        [{"uid": "a", "score": 0.6}, {"uid": "b", "score": 0.5}],
        [{"uid": "b", "column": "total", "score": 0.9}, {"uid": "b", "column": "x", "score": 0.7}, {"uid": "c", "column": "y", "score": 0.4}],
        k=10,
    )
    assert [(s.uid, s.score, s.matched_on) for s in seeds] == [
        ("b", 0.9, "column:total"),
        ("a", 0.6, "table"),
        ("c", 0.4, "column:y"),
    ]


def test_rollup_table_wins_ties_and_uid_breaks_ties():
    seeds = retrieve.rollup([{"uid": "z", "score": 0.5}, {"uid": "a", "score": 0.5}], [{"uid": "z", "column": "c", "score": 0.5}], k=5)
    assert [(s.uid, s.matched_on) for s in seeds] == [("a", "table"), ("z", "table")]


def test_rollup_limits_k_and_rounds():
    hits = [{"uid": f"t{i}", "score": 0.123456 + i / 100} for i in range(5)]
    seeds = retrieve.rollup(hits, [], k=2)
    assert [s.uid for s in seeds] == ["t4", "t3"] and seeds[0].score == 0.1635
    assert retrieve.rollup(hits, [], k=0) == []


# --------------------------------------------------------------------------- embed + retrieve


@pytest.fixture()
def store(meta):
    s = FakeVectorStore()
    s.add_metadata(meta)
    # a second database that shares table names, to prove db_id scoping
    other = meta.model_copy(deep=True)
    for t in other.tables:
        t.uid = t.uid.replace(f"{DB}:", "other:", 1)
    for c in other.columns:
        c.uid = c.uid.replace(f"{DB}:", "other:", 1)
        c.table_uid = c.table_uid.replace(f"{DB}:", "other:", 1)
    s.add_metadata(other, db_id="other")
    return s


@pytest.fixture()
def embedded_both(store, meta, monkeypatch):
    """Both databases embedded, so the global index really holds nodes of each."""
    other = meta.model_copy(deep=True)
    for t in other.tables:
        t.uid = t.uid.replace(f"{DB}:", "other:", 1)
    for c in other.columns:
        c.uid = c.uid.replace(f"{DB}:", "other:", 1)
        c.table_uid = c.table_uid.replace(f"{DB}:", "other:", 1)
    emb = WordEmbedder()
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta if db_id == DB else other))
    for db in (DB, "other"):
        enrich.embed_nodes(db, emb, store)
    return store, emb


@pytest.fixture()
def embedded(store, meta, monkeypatch):
    emb = WordEmbedder()
    details = queries.assemble_details  # noqa: F841  (queries is exercised through get_all_tables below)
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    for db in (DB,):
        enrich.embed_nodes(db, emb, store)
    return store, emb


def _details(meta):
    from schemagraph.models import TableDetail

    by_table = {}
    for c in meta.columns:
        by_table.setdefault(c.table_uid, []).append(c)
    return [TableDetail(table=t, columns=by_table.get(t.uid, [])) for t in meta.tables]


def test_embed_nodes_stores_a_vector_on_every_node(embedded, meta):
    store, emb = embedded
    mine_t = [p for p in store.tables.values() if p["db_id"] == DB]
    mine_c = [p for p in store.columns.values() if p["db_id"] == DB]
    assert all(len(p["embedding"]) == DIM for p in mine_t + mine_c)
    assert len(mine_t) == len(meta.tables) and len(mine_c) == len(meta.columns)
    # the other database was not touched
    assert not any("embedding" in p for p in store.tables.values() if p["db_id"] == "other")


def test_embed_nodes_report_and_texts(store, meta, monkeypatch):
    emb = WordEmbedder()
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    report = enrich.embed_nodes(DB, emb, store)
    assert (report.tables, report.columns) == (len(meta.tables), len(meta.columns))
    assert sum(len(c) for c in emb.document_calls) == report.tables + report.columns


def test_embed_nodes_rejects_wrong_vector_count(store, meta, monkeypatch):
    class Short(WordEmbedder):
        def embed_documents(self, texts):
            return super().embed_documents(texts)[:-1]

    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(meta))
    with pytest.raises(vector.EmbeddingError):
        enrich.embed_nodes(DB, Short(), store)


def test_retrieve_finds_table_by_its_name(embedded):
    store, emb = embedded
    seeds = retrieve.retrieve_seed_tables(DB, "salesorderheader", 3, embedder=emb, session=store)
    assert seeds[0].uid == f"{DB}:sales.salesorderheader"
    assert len(seeds) == 3 and all(s.uid.startswith(f"{DB}:") for s in seeds)
    assert seeds == sorted(seeds, key=lambda s: (-s.score, s.uid))


def test_retrieve_finds_table_through_a_column(embedded, meta):
    store, emb = embedded
    col = next(c for c in meta.columns if c.name == "totaldue")
    seeds = retrieve.retrieve_seed_tables(DB, "totaldue", 5, embedder=emb, session=store)
    hit = next(s for s in seeds if s.uid == col.table_uid)
    assert hit.matched_on == "column:totaldue"


def test_retrieve_is_scoped_to_db_id_even_when_other_db_fills_the_index(embedded_both):
    store, emb = embedded_both
    # other-db nodes tie with ours; with a tiny candidate pool they crowd ours out of the global top-n
    store.queries.clear()
    seeds = retrieve.retrieve_seed_tables(DB, "customer", 5, embedder=emb, session=store, candidates=2)
    assert all(s.uid.startswith(f"{DB}:") for s in seeds)
    assert all(p["db_id"] == DB for _, p in store.queries if "db_id" in p)


def test_retrieve_fewer_than_k_and_min_score_and_empty(embedded, meta):
    store, emb = embedded
    assert len(retrieve.retrieve_seed_tables(DB, "customer", 1000, embedder=emb, session=store)) <= len(meta.tables)
    assert retrieve.retrieve_seed_tables(DB, "customer", 5, embedder=emb, session=store, min_score=1.01) == []
    assert retrieve.retrieve_seed_tables(DB, "   ", 5, embedder=emb, session=store) == []
    assert retrieve.retrieve_seed_tables(DB, "customer", 0, embedder=emb, session=store) == []
    assert emb.query_calls.count("   ") == 0  # no embedding call for an empty question


def test_retrieve_embeds_the_question_once(embedded):
    store, emb = embedded
    emb.query_calls.clear()
    retrieve.retrieve_seed_tables(DB, "customer orders", 3, embedder=emb, session=store)
    assert emb.query_calls == ["customer orders"]


# --------------------------------------------------------------------------- HTTP embedder


class Handler(BaseHTTPRequestHandler):
    script: list = []   # (status, body) per request, in order
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        status, payload = Handler.script.pop(0) if Handler.script else (200, None)
        if payload is None:
            data = [{"index": i, "embedding": [float(i)] * 3} for i in range(len(body["input"]))]
            payload = {"data": data[::-1]}  # out of order on purpose
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    Handler.script, Handler.seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def make(url, monkeypatch, **kw):
    monkeypatch.setenv("TEST_EMB_KEY", "sekret")
    cfg = EmbeddingConfig(base_url=url, model="m", dimensions=3, api_key_env="TEST_EMB_KEY", **kw)
    sleeps = []
    return vector.OpenAICompatibleEmbedder(cfg, sleep=sleeps.append), sleeps


def test_http_embedder_batches_orders_and_authenticates(server, monkeypatch):
    emb, _ = make(server, monkeypatch, batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert out == [[0.0] * 3, [1.0] * 3, [0.0] * 3]  # sorted back by index, per batch
    assert [len(b["input"]) for _, _, b in Handler.seen] == [2, 1]
    assert all(p == "/v1/embeddings" and a == "Bearer sekret" and b["model"] == "m" for p, a, b in Handler.seen)
    assert all("dimensions" not in b for _, _, b in Handler.seen)


def test_http_embedder_sends_dimensions_when_asked(server, monkeypatch):
    emb, _ = make(server, monkeypatch, send_dimensions=True)
    emb.embed_query("x")
    assert Handler.seen[0][2]["dimensions"] == 3


def test_http_embedder_retries_then_succeeds(server, monkeypatch):
    Handler.script = [(429, {"error": "slow down"}), (503, {"error": "x"})]
    emb, sleeps = make(server, monkeypatch)
    assert emb.embed_query("x") == [0.0] * 3
    assert len(Handler.seen) == 3 and sleeps == [5, 1]   # 429 waits on the rate-limit schedule, 503 on the short one


def test_http_embedder_does_not_retry_client_errors(server, monkeypatch):
    Handler.script = [(401, {"error": "bad key"})]
    emb, sleeps = make(server, monkeypatch)
    with pytest.raises(vector.EmbeddingError, match="401"):
        emb.embed_query("x")
    assert len(Handler.seen) == 1 and sleeps == []


def test_http_embedder_gives_up_after_retries(server, monkeypatch):
    Handler.script = [(500, {})] * 10
    emb, _ = make(server, monkeypatch)
    with pytest.raises(vector.EmbeddingError, match="500"):
        emb.embed_query("x")
    assert len(Handler.seen) == 4


def test_http_embedder_checks_dimensions(server, monkeypatch):
    Handler.script = [(200, {"data": [{"index": 0, "embedding": [1.0, 2.0]}]})]
    emb, _ = make(server, monkeypatch)
    with pytest.raises(vector.EmbeddingError, match="2-dimensional"):
        emb.embed_query("x")


def test_http_embedder_rejects_malformed_body(server, monkeypatch):
    Handler.script = [(200, {"oops": 1})]
    emb, _ = make(server, monkeypatch)
    with pytest.raises(vector.EmbeddingError, match="unexpected response"):
        emb.embed_query("x")


def test_http_embedder_unreachable(monkeypatch):
    emb, sleeps = make("http://127.0.0.1:9/v1", monkeypatch)
    with pytest.raises(vector.EmbeddingError, match="cannot reach"):
        emb.embed_query("x")
    assert len(sleeps) == 3
