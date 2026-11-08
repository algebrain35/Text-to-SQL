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
