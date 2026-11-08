"""Two ways to get the tables a question never names into the context: neighbours in the embedded table
text (`include_relationships`), and adding the tables the seeds reference before planning (`expand_seeds`)."""

from types import SimpleNamespace

from schemagraph.graph import enrich, retrieve, vector
from schemagraph.models import (
    ColumnMeta,
    JoinPlan,
    RelationshipMeta,
    SeedTable,
    TableDetail,
    TableMeta,
)
from schemagraph.pipeline import context as context_mod
from schemagraph.pipeline.explain import explain_context

DB = "demo"


def uid(table):
    return f"{DB}:{table}"


def seed(table, score):
    return SeedTable(uid=uid(table), score=score, matched_on="table")


def edge(src, dst, confidence=1.0):
    return SimpleNamespace(from_table=uid(src), to_table=uid(dst), confidence=confidence)


# --- expand_seeds ------------------------------------------------------------------------------


def test_adds_the_tables_a_seed_references_and_labels_them():
    out = retrieve.expand_seeds([seed("s.salesperson", 0.85)], [edge("s.salesperson", "p.person")], max_added=2)
    assert [s.uid for s in out] == [uid("s.salesperson"), uid("p.person")]
    added = out[1]
    assert added.matched_on == "reference_of:s.salesperson" and added.score == 0.80


def test_children_are_not_added_only_what_the_seeds_reference():
    out = retrieve.expand_seeds([seed("p.person", 0.8)], [edge("s.salesperson", "p.person")], max_added=3)
    assert [s.uid for s in out] == [uid("p.person")]


def test_shared_parents_come_first_then_fewer_links_then_better_score():
    seeds = [seed("a", 0.9), seed("b", 0.8)]
    edges = [edge("a", "only_a"), edge("a", "shared"), edge("b", "shared"), edge("b", "only_b")]
    out = retrieve.expand_seeds(seeds, edges, max_added=2)
    assert [s.uid.split(":")[1] for s in out[2:]] == ["shared", "only_a"]  # shared by both; then the better seed's


def test_hops_reach_further_and_closer_tables_rank_first():
    edges = [edge("a", "b"), edge("b", "c")]
    one = retrieve.expand_seeds([seed("a", 0.9)], edges, max_added=5, hops=1)
    two = retrieve.expand_seeds([seed("a", 0.9)], edges, max_added=5, hops=2)
    assert [s.uid.split(":")[1] for s in one] == ["a", "b"]
    assert [s.uid.split(":")[1] for s in two] == ["a", "b", "c"]
    assert two[2].score == 0.80 and two[1].score == 0.85


def test_weak_inferred_links_self_references_existing_seeds_and_zero_budget_add_nothing():
    seeds = [seed("a", 0.9), seed("b", 0.8)]
    assert retrieve.expand_seeds(seeds, [edge("a", "weak", confidence=0.6)], max_added=3) == seeds
    assert retrieve.expand_seeds(seeds, [edge("a", "a")], max_added=3) == seeds
    assert retrieve.expand_seeds(seeds, [edge("a", "b")], max_added=3) == seeds     # already a seed
    assert retrieve.expand_seeds(seeds, [edge("a", "c")], max_added=0) == seeds
    assert retrieve.expand_seeds([], [edge("a", "c")], max_added=3) == []


def test_expansion_is_deterministic_whatever_the_edge_order():
    seeds = [seed("a", 0.9)]
    edges = [edge("a", "z"), edge("a", "m"), edge("a", "c")]
    assert retrieve.expand_seeds(seeds, edges, max_added=2) == retrieve.expand_seeds(seeds, edges[::-1], max_added=2)


# --- build_context -----------------------------------------------------------------------------


def test_build_context_expands_only_when_asked_and_only_in_graph_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr(context_mod.retrieve, "retrieve_seed_tables", lambda db, q, k: [seed("s.salesperson", 0.85)])
    monkeypatch.setattr(context_mod.queries, "get_reference_edges",
                        lambda db, include_inferred=True: [edge("s.salesperson", "p.person")])

    def fake_plan(db, seeds, **opts):
        seen["seeds"] = [s.uid for s in seeds]
        return JoinPlan(db_id=db, status="none", algorithm="none", seed_tables=seeds, tables=[s.uid for s in seeds])

    monkeypatch.setattr(context_mod.planner, "plan_joins", fake_plan)
    monkeypatch.setattr(context_mod.queries, "get_tables", lambda db, uids: [])

    _, seeds = context_mod.build_context(DB, "q", "graph", 5)
    assert seen["seeds"] == [uid("s.salesperson")] and len(seeds) == 1

    _, seeds = context_mod.build_context(DB, "q", "graph", 5, expand={"max_added": 2, "hops": 1})
    assert seen["seeds"] == [uid("s.salesperson"), uid("p.person")] and len(seeds) == 2

    seen.clear()
    _, seeds = context_mod.build_context(DB, "q", "retrieval", 5, expand={"max_added": 2, "hops": 1})
    assert "seeds" not in seen and len(seeds) == 1       # retrieval mode is the no-graph baseline


def test_expand_options_follows_the_pipeline_config():
    assert context_mod.expand_options(None) is None
    assert context_mod.expand_options(SimpleNamespace(expand_seeds=0, expand_hops=2)) is None
    assert context_mod.expand_options(SimpleNamespace(expand_seeds=3, expand_hops=2)) == {"max_added": 3, "hops": 2}


def test_the_explanation_says_why_an_added_table_is_there():
    from schemagraph.models import SchemaContext

    added = SeedTable(uid=uid("p.person"), score=0.8, matched_on="reference_of:s.salesperson")
    ctx = SchemaContext(db_id=DB, mode="graph", tables=[added.uid], ddl="", join_text="", join_plan=None, notes=[])
    reason = explain_context(ctx, [added]).tables[0].reason
    assert reason == "Added because s.salesperson has a foreign key to it."


# --- relationships in the embedded text --------------------------------------------------------


def detail(table, refs=(), confidence=1.0):
    t = uid(f"s.{table}")
    col = ColumnMeta(uid=f"{t}.id", table_uid=t, name="id", data_type="integer", ordinal_position=1,
                     is_nullable=False, is_primary_key=True)
    rels = [RelationshipMeta(from_column=f"{t}.id", to_column=f"{uid('s.' + r)}.id", source="declared",
                             confidence=confidence, cardinality="N:1") for r in refs]
    return TableDetail(table=TableMeta(uid=t, schema="s", name=table, description=f"The {table}."),
                       columns=[col], references=rels)


def test_table_text_lists_neighbours_after_the_description_and_caps_them():
    t = detail("product").table
    assert vector.table_text(t) == "s.product: The product."
    text = vector.table_text(t, references=["s.subcategory"], referenced_by=["s.detail", "s.vendor"])
    assert text == "s.product: The product. References: s.subcategory. Referenced by: s.detail, s.vendor."
    many = [f"s.t{i:02d}" for i in range(20)]
    capped = vector.table_text(t, referenced_by=many)
    assert capped.endswith("s.t11, and 8 more.") and "s.t12" not in capped


def test_embedding_inputs_add_neighbours_only_when_asked():
    details = [detail("detail", refs=["product"]), detail("product", refs=["subcategory"]), detail("subcategory")]
    plain, _ = enrich.embedding_inputs(details)
    rich, _ = enrich.embedding_inputs(details, include_relationships=True)
    assert dict(plain)[uid("s.product")] == "s.product: The product."
    assert dict(rich)[uid("s.product")] == (
        "s.product: The product. References: s.subcategory. Referenced by: s.detail."
    )
    assert dict(rich)[uid("s.detail")].endswith("References: s.product.")
    assert dict(rich)[uid("s.subcategory")].endswith("Referenced by: s.product.")


def test_weak_links_and_self_references_stay_out_of_the_text():
    details = [detail("a", refs=["b"], confidence=0.5), detail("b", refs=["b"])]
    rich, _ = enrich.embedding_inputs(details, include_relationships=True)
    assert dict(rich)[uid("s.a")] == "s.a: The a." and dict(rich)[uid("s.b")] == "s.b: The b."


def test_a_changed_text_changes_the_fingerprint_so_it_is_re_embedded():
    emb = SimpleNamespace(config=SimpleNamespace(provider="p", model="m", dimensions=3, send_dimensions=False))
    assert enrich.fingerprint(emb, "s.product: The product.") != enrich.fingerprint(
        emb, "s.product: The product. Referenced by: s.detail."
    )
