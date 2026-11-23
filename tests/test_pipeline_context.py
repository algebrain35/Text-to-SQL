from schemagraph.models import (
    ColumnMeta,
    JoinPlan,
    PlannedJoin,
    RelationshipMeta,
    TableDetail,
    TableMeta,
)
from schemagraph.pipeline.context import render_ddl, render_joins


def _column(uid, table_uid, name, *, nullable=True, primary=False):
    return ColumnMeta(
        uid=uid,
        table_uid=table_uid,
        name=name,
        data_type="integer",
        ordinal_position=1,
        is_nullable=nullable,
        is_primary_key=primary,
    )


def test_render_ddl_quotes_names_and_adds_primary_and_foreign_keys():
    parent_uid = "demo:public.parent"
    child_uid = "demo:public.child"
    parent = TableDetail(
        table=TableMeta(uid=parent_uid, schema="public", name='parent"table', primary_key=["id"]),
        columns=[_column(f"{parent_uid}.id", parent_uid, "id", nullable=False, primary=True)],
        references=[],
    )
    child = TableDetail(
        table=TableMeta(uid=child_uid, schema="public", name="child"),
        columns=[_column(f"{child_uid}.parent_id", child_uid, "parent_id")],
        references=[
            RelationshipMeta(
                from_column=f"{child_uid}.parent_id",
                to_column=f"{parent_uid}.id",
                source="declared",
                confidence=1.0,
                cardinality="N:1",
                constraint_name="child_parent_fk",
            )
        ],
    )

    ddl = render_ddl([parent, child])

    assert 'CREATE TABLE "public"."parent""table"' in ddl
    assert 'PRIMARY KEY ("id")' in ddl
    assert 'FOREIGN KEY ("parent_id") REFERENCES "public"."parent""table" ("id")' in ddl


def test_render_joins_discloses_inferred_relationship_confidence():
    plan = JoinPlan(
        db_id="demo",
        status="full",
        algorithm="shortest_path",
        joins=[
            PlannedJoin(
                from_column="demo:public.child.parent_id",
                to_column="demo:public.parent.id",
                source="inferred",
                confidence=0.8,
                cardinality="N:1",
                traversal="many_to_one",
                weight=1.2,
            )
        ],
    )

    assert "inferred, confidence 0.80" in render_joins(plan)


def _parent_and_child(source):
    parent_uid, child_uid = "demo:public.parent", "demo:public.child"
    parent = TableDetail(
        table=TableMeta(uid=parent_uid, schema="public", name="parent", primary_key=["id"]),
        columns=[_column(f"{parent_uid}.id", parent_uid, "id", nullable=False, primary=True)],
        references=[],
    )
    child = TableDetail(
        table=TableMeta(uid=child_uid, schema="public", name="child"),
        columns=[_column(f"{child_uid}.parent_id", child_uid, "parent_id")],
        references=[
            RelationshipMeta(
                from_column=f"{child_uid}.parent_id",
                to_column=f"{parent_uid}.id",
                source=source,
                confidence=0.85 if source == "inferred" else 1.0,
                cardinality="N:1",
                constraint_name=None if source == "inferred" else "child_parent_fk",
            )
        ],
    )
    return [parent, child]


def test_render_ddl_marks_an_inferred_relationship_as_a_guess():
    ddl = render_ddl(_parent_and_child("inferred"))

    assert "FOREIGN KEY" not in ddl
    assert (
        '-- Inferred relationship, not a declared foreign key (confidence 0.85): '
        '"public"."child"."parent_id" -> "public"."parent"."id"'
    ) in ddl
    assert ddl.rstrip().endswith('"public"."parent"."id"')  # after the statement, not inside it


def test_render_ddl_leaves_inferred_relationships_out_when_they_are_turned_off():
    ddl = render_ddl(_parent_and_child("inferred"), include_inferred=False)
    assert "Inferred" not in ddl and "FOREIGN KEY" not in ddl
    # declared keys are unaffected
    assert "FOREIGN KEY" in render_ddl(_parent_and_child("declared"), include_inferred=False)


# --- build_context ------------------------------------------------------------------------------

from schemagraph.models import SeedTable  # noqa: E402
from schemagraph.pipeline import context as context_mod  # noqa: E402


def _fake_backends(monkeypatch, plan_status, *, unreachable=(), details=None):
    asked = {}
    seeds = [SeedTable(uid=f"demo:public.{n}", score=s, matched_on="table")
             for n, s in (("child", 0.9), ("parent", 0.8), ("island", 0.7))]
    monkeypatch.setattr(context_mod.retrieve, "retrieve_seed_tables", lambda db, q, k: seeds)

    def fake_plan(db, seeds, **opts):
        asked["planner_options"] = opts
        return JoinPlan(db_id=db, status=plan_status, algorithm="shortest_path", seed_tables=seeds,
                        tables=["demo:public.child", "demo:public.parent"], unreachable_seeds=list(unreachable))

    def fake_get_tables(db, uids):
        asked["uids"] = list(uids)
        return details or []

    monkeypatch.setattr(context_mod.planner, "plan_joins", fake_plan)
    monkeypatch.setattr(context_mod.queries, "get_tables", fake_get_tables)
    return asked


def test_build_context_keeps_unreachable_seeds_with_a_note(monkeypatch):
    asked = _fake_backends(monkeypatch, "partial", unreachable=["demo:public.island"])

    context, _ = context_mod.build_context("demo", "q", "graph", 5)

    assert asked["uids"] == ["demo:public.child", "demo:public.parent", "demo:public.island"]
    assert len(context.notes) == 1
    assert "public.island" in context.notes[0] and "No known relationship" in context.notes[0]


def test_build_context_adds_no_note_when_every_seed_is_connected(monkeypatch):
    asked = _fake_backends(monkeypatch, "full")

    context, _ = context_mod.build_context("demo", "q", "graph", 5)

    assert asked["uids"] == ["demo:public.child", "demo:public.parent"]
    assert context.notes == []


def test_build_context_passes_include_inferred_to_the_ddl(monkeypatch):
    _fake_backends(monkeypatch, "full", details=_parent_and_child("inferred"))

    with_inferred, _ = context_mod.build_context("demo", "q", "graph", 5)
    without, _ = context_mod.build_context("demo", "q", "graph", 5, planner_options={"include_inferred": False})

    assert "Inferred relationship" in with_inferred.ddl
    assert "Inferred relationship" not in without.ddl


def test_planner_defaults_follow_the_pipeline_config():
    from types import SimpleNamespace

    assert context_mod.planner_defaults(None) == {}
    assert context_mod.planner_defaults(SimpleNamespace(planner_selection="all", prize_scale=3.0)) == {}
    assert context_mod.planner_defaults(SimpleNamespace(planner_selection="prize", prize_scale=4.5)) == {
        "selection": "prize", "prize_scale": 4.5,
    }
