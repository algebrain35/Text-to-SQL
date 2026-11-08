"""enrich.embed_nodes saves work as it goes and resumes: chunked writes, tables first, rate limiting
that stops cleanly, and a rerun that embeds only what is missing or changed."""

from types import SimpleNamespace

import pytest

from schemagraph.graph import enrich, vector
from tests.test_retrieve_unit import DB, FakeVectorStore, WordEmbedder, _details, meta, store  # noqa: F401


@pytest.fixture()
def graph(store, meta, monkeypatch):  # noqa: F811
    """A store holding the sample database, and a way to change what the graph says about it."""
    state = SimpleNamespace(meta=meta)
    monkeypatch.setattr(enrich.queries, "get_all_tables", lambda db_id, s: _details(state.meta))
    state.store = store
    return state


class LimitedEmbedder(WordEmbedder):
    """Works for `allow` calls to embed_documents, then is rate limited."""

    def __init__(self, allow, config=None):
        super().__init__()
        self.allow, self.calls = allow, 0
        if config is not None:
            self.config = config

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls > self.allow:
            raise vector.RateLimited("rate limited (HTTP 429) and the 120s wait budget (max_wait_s) is used up")
        return super().embed_documents(texts)


def embedded_texts(emb):
    return sum(len(c) for c in emb.document_calls)


def stored_count(store):
    return (
        sum("embedding" in p for p in store.tables.values()),
        sum("embedding" in p for p in store.columns.values()),
    )


def test_work_is_written_in_chunks_with_tables_first(graph, meta):
    emb = WordEmbedder()
    report = enrich.embed_nodes(DB, emb, graph.store, chunk_size=10)
    assert report.complete and (report.tables, report.columns) == (len(meta.tables), len(meta.columns))
    assert all(len(call) <= 10 for call in emb.document_calls)
    assert graph.store.writes == [len(meta.tables)] + [10] * (len(meta.columns) // 10) + [len(meta.columns) % 10]
    first_set = next(q for q, _ in graph.store.queries if q.startswith("UNWIND"))
    assert "(n:Table" in first_set


def test_rate_limit_stops_cleanly_and_keeps_what_was_saved(graph, meta):
    emb = LimitedEmbedder(allow=3)
    report = enrich.embed_nodes(DB, emb, graph.store, chunk_size=5)
    assert not report.complete and "rate limited" in report.stopped
    assert report.tables == len(meta.tables)                               # tables first: all of them made it
    assert 0 < report.columns < len(meta.columns)                          # some columns saved, not all
    assert stored_count(graph.store) == (report.tables, report.columns)    # what the report says is what is saved
    assert report.tables_current == report.tables and report.columns_current == report.columns
    assert (report.tables_total, report.columns_total) == (len(meta.tables), len(meta.columns))


def test_a_rerun_embeds_only_the_rest_then_nothing(graph, meta):
    first = enrich.embed_nodes(DB, LimitedEmbedder(allow=3), graph.store, chunk_size=5)
    healthy = WordEmbedder()
    second = enrich.embed_nodes(DB, healthy, graph.store, chunk_size=5)
    assert second.complete and second.tables == 0
    assert second.columns == len(meta.columns) - first.columns
    assert embedded_texts(healthy) == second.columns                       # nothing was embedded twice
    assert second.tables_current == len(meta.tables) and second.columns_current == len(meta.columns)
    assert stored_count(graph.store) == (len(meta.tables), len(meta.columns))

    third = WordEmbedder()
    report = enrich.embed_nodes(DB, third, graph.store, chunk_size=5)
    assert third.document_calls == [] and (report.tables, report.columns) == (0, 0) and report.complete


def test_a_changed_description_is_embedded_again_and_only_that(graph, meta):
    enrich.embed_nodes(DB, WordEmbedder(), graph.store)
    graph.meta = meta.model_copy(deep=True)
    column = graph.meta.columns[3]
    column.description = "a brand new description of this column"
    emb = WordEmbedder()
    report = enrich.embed_nodes(DB, emb, graph.store)
    assert (report.tables, report.columns) == (0, 1) and embedded_texts(emb) == 1
    assert "brand new description" in emb.document_calls[0][0]


def test_a_different_model_under_the_same_name_is_embedded_again(graph, meta):
    def with_model(model):
        return LimitedEmbedder(allow=99, config=SimpleNamespace(provider="x", model=model, dimensions=256, send_dimensions=False))

    enrich.embed_nodes(DB, with_model("one"), graph.store)
    same = with_model("one")
    assert enrich.embed_nodes(DB, same, graph.store).tables == 0 and same.document_calls == []
    other = with_model("two")
    report = enrich.embed_nodes(DB, other, graph.store)
    assert (report.tables, report.columns) == (len(meta.tables), len(meta.columns))


def test_reembed_forces_everything(graph, meta):
    enrich.embed_nodes(DB, WordEmbedder(), graph.store)
    emb = WordEmbedder()
    report = enrich.embed_nodes(DB, emb, graph.store, reembed=True)
    assert (report.tables, report.columns) == (len(meta.tables), len(meta.columns)) and embedded_texts(emb) == report.tables + report.columns


def test_a_vector_without_a_fingerprint_is_replaced(graph, meta):
    """Vectors written before fingerprints existed cannot be trusted to match their text."""
    for node in list(graph.store.tables.values()) + list(graph.store.columns.values()):
        node["embedding"] = [0.0] * 256
    report = enrich.embed_nodes(DB, WordEmbedder(), graph.store)
    assert (report.tables, report.columns) == (len(meta.tables), len(meta.columns))


def test_fingerprints_are_stored_per_vector_name(graph, meta):
    enrich.embed_nodes(DB, WordEmbedder(), graph.store)
    enrich.embed_nodes(DB, WordEmbedder(), graph.store, name="other")
    node = next(iter(graph.store.tables.values()))
    assert "embedding_hash" in node and "embedding_other_hash" in node
    assert all(len(v) == 256 for v in (node["embedding"], node["embedding_other"]))


def test_other_errors_are_raised_but_earlier_chunks_stay_saved(graph, meta):
    class Broken(WordEmbedder):
        def embed_documents(self, texts):
            if self.document_calls:
                raise vector.EmbeddingError("HTTP 400 bad request")
            return super().embed_documents(texts)

    with pytest.raises(vector.EmbeddingError, match="400"):
        enrich.embed_nodes(DB, Broken(), graph.store, chunk_size=5)
    assert stored_count(graph.store)[0] == 5


def test_a_database_with_nothing_to_do_makes_no_embedding_calls(graph, meta):
    enrich.embed_nodes(DB, WordEmbedder(), graph.store)
    emb = LimitedEmbedder(allow=0)                                         # would be rate limited if asked anything
    assert enrich.embed_nodes(DB, emb, graph.store).complete and emb.calls == 0
