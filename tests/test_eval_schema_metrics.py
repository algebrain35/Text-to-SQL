from schemagraph.eval.schema_metrics import bridge_tables_needed, schema_precision, schema_recall, tables_in_sql


def test_tables_in_sql_qualified_and_unqualified():
    sql = "SELECT * FROM Sales.Customer c JOIN person.person p ON 1=1 JOIN orders o ON 1=1"
    assert tables_in_sql(sql) == {"sales.customer", "person.person", "public.orders"}
    assert tables_in_sql("SELECT * FROM orders", default_schema="shop") == {"shop.orders"}


def test_tables_in_sql_ignores_ctes_and_survives_bad_sql():
    sql = "WITH t AS (SELECT 1 FROM sales.customer) SELECT * FROM t"
    assert tables_in_sql(sql) == {"sales.customer"}
    assert tables_in_sql("SELECT * FROM") == set()


def test_schema_recall():
    gold = {"sales.customer", "sales.salesorderheader", "person.person"}
    assert schema_recall(gold, gold | {"extra.table"}) == 1.0
    assert schema_recall(gold, {"sales.customer"}) == 1 / 3
    assert schema_recall(gold, set()) == 0.0
    assert schema_recall(set(), {"anything"}) == 1.0


def test_bridge_tables_needed():
    gold = {"sales.customer", "sales.salesorderheader", "person.person"}
    seeds = {"sales.salesorderheader", "person.person", "other.table"}
    assert bridge_tables_needed(gold, seeds) == {"sales.customer"}
    assert bridge_tables_needed(gold, gold) == set()


def test_schema_precision():
    gold = {"sales.salesorderheader", "sales.customer"}
    assert schema_precision(gold, gold) == 1.0
    assert schema_precision(gold, gold | {"x.a", "x.b"}) == 0.5
    assert schema_precision(gold, {"x.a"}) == 0.0
    assert schema_precision(gold, set()) is None
