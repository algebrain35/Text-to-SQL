import pytest

from schemagraph.models import ExplainResult
from schemagraph.pipeline import validate as validate_module
from schemagraph.sql_validation import check_select, is_single_select, referenced_tables


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 1",
        "SELECT * FROM customers WHERE id = 1;",
        "WITH totals AS (SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id) "
        "SELECT * FROM totals",
        "SELECT id FROM customers UNION SELECT customer_id FROM orders",
        "-- query comment\nSELECT 'DELETE FROM customers' AS example",
    ],
)
def test_is_single_select_accepts_readonly_queries(statement):
    assert is_single_select(statement) is True


@pytest.mark.parametrize(
    "statement",
    [
        "",
        "SELECT * FROM",
        "INSERT INTO customers VALUES (1)",
        "UPDATE customers SET name = 'x'",
        "DELETE FROM customers",
        "SELECT 1; SELECT 2",
        "SELECT * INTO copied_customers FROM customers",
        "WITH changed AS (DELETE FROM customers RETURNING *) SELECT * FROM changed",
        "SELECT * FROM customers FOR UPDATE",
    ],
)
def test_is_single_select_rejects_unsafe_or_invalid_sql(statement):
    assert is_single_select(statement) is False


def test_is_single_select_rejects_missing_sql():
    assert is_single_select(None) is False


def test_validate_rejects_unsafe_sql_without_explaining(monkeypatch):
    def unexpected_explain(*args, **kwargs):
        pytest.fail("Unsafe SQL must be rejected before EXPLAIN")

    monkeypatch.setattr(validate_module.db, "explain", unexpected_explain)

    result = validate_module.validate("adventureworks", "DELETE FROM customers")

    assert result.ok is False
    assert result.error == "Only a single read-only SELECT statement is allowed: found DELETE."


def test_validate_explains_a_safe_select(monkeypatch):
    calls = []

    def fake_explain(db_id, statement):
        calls.append((db_id, statement))
        return ExplainResult(ok=True)

    monkeypatch.setattr(validate_module.db, "explain", fake_explain)

    result = validate_module.validate("adventureworks", "SELECT 1")

    assert result.ok is True
    assert calls == [("adventureworks", "SELECT 1")]


# --- check_select: reasons, parentheses, denied functions -------------------------


@pytest.mark.parametrize(
    "statement",
    ["(SELECT 1)", "((SELECT 1))", "(SELECT 1) ORDER BY 1", "(SELECT 1) UNION (SELECT 2)"],
)
def test_parenthesized_selects_are_accepted(statement):
    assert check_select(statement) == (True, None)


@pytest.mark.parametrize(
    "statement,fragment",
    [
        (None, "empty"),
        ("   ", "empty"),
        ("SELECT * FROM", "parsed"),
        ("SELECT 1; SELECT 2", "2 statements"),
        ("DELETE FROM customers", "found DELETE"),
        ("INSERT INTO t VALUES (1)", "found INSERT"),
        ("WITH c AS (DELETE FROM t RETURNING *) SELECT * FROM c", "DELETE is not allowed"),
        ("SELECT * INTO copy FROM t", "INTO"),
        ("SELECT * FROM t FOR UPDATE", "FOR UPDATE"),
    ],
)
def test_check_select_gives_a_reason(statement, fragment):
    check = check_select(statement)
    assert check.ok is False
    assert fragment in check.reason


@pytest.mark.parametrize(
    "statement,name",
    [
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT pg_sleep(100)", "pg_sleep"),
        ("SELECT pg_sleep_for('5 minutes')", "pg_sleep_for"),
        ("SELECT pg_catalog.pg_sleep(1)", "pg_sleep"),
        ('SELECT "pg_sleep"(1)', "pg_sleep"),
        ("SELECT PG_SLEEP(1)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT * FROM pg_ls_dir('.')", "pg_ls_dir"),
        ("SELECT lo_import('/etc/passwd')", "lo_import"),
        ("SELECT * FROM dblink('host=x', 'select 1') AS t(a int)", "dblink"),
        ("SELECT pg_terminate_backend(1)", "pg_terminate_backend"),
        ("SELECT pg_cancel_backend(1)", "pg_cancel_backend"),
        ("SELECT pg_advisory_lock(1)", "pg_advisory_lock"),
        ("SELECT nextval('s')", "nextval"),
        ("SELECT setval('s', 1)", "setval"),
        ("SELECT pg_notify('c', 'x')", "pg_notify"),
        ("SELECT query_to_xml('select 1', true, false, '')", "query_to_xml"),
        ("WITH c AS (SELECT pg_sleep(1)) SELECT * FROM c", "pg_sleep"),
        ("SELECT * FROM (SELECT nextval('s')) x", "nextval"),
    ],
)
def test_denied_functions_are_rejected(statement, name):
    check = check_select(statement)
    assert check.ok is False
    assert f"{name}()" in check.reason


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT lower(name), count(*), sum(amount), coalesce(a, 0), now() FROM t GROUP BY 1",
        "SELECT date_trunc('month', d), generate_series(1, 3) FROM t",
        "SELECT 'pg_sleep(1) and set_config' AS note",   # text, not a call
        "SELECT 1 AS lo_import",                          # alias, not a call
        "SELECT x FROM t -- ; DROP TABLE t",              # comment, not a statement
        "SELECT lo_total FROM t",                         # column that merely starts with lo_
    ],
)
def test_ordinary_selects_with_lookalikes_pass(statement):
    assert check_select(statement).ok is True


def test_every_rejection_has_a_reason():
    for statement in ["", "SELECT 1; SELECT 2", "DELETE FROM t", "SELECT pg_sleep(1)", "SELECT * FROM t FOR UPDATE"]:
        check = check_select(statement)
        assert check.ok is False and check.reason


# --- referenced_tables --------------------------------------------------------------


def test_referenced_tables_qualified_unqualified_and_lowercase():
    sql = "SELECT * FROM Sales.Customer c JOIN person.person p ON 1=1 JOIN orders o ON 1=1"
    assert referenced_tables(sql) == {"sales.customer", "person.person", "orders"}


def test_referenced_tables_ignores_ctes_functions_and_subqueries_inside():
    sql = (
        "WITH totals AS (SELECT customerid FROM sales.salesorderheader) "
        "SELECT * FROM totals JOIN generate_series(1, 3) g ON true "
        "WHERE EXISTS (SELECT 1 FROM person.person)"
    )
    assert referenced_tables(sql) == {"sales.salesorderheader", "person.person"}


def test_referenced_tables_of_unparsable_sql_is_empty():
    assert referenced_tables("SELECT * FROM") == set()


# An unterminated quote fails in sqlglot's tokenizer, which raises TokenError (not ParseError).
UNTOKENIZABLE = ["SELECT 'abc FROM t", 'SELECT "a FROM t', "SELECT 1 FROM t WHERE a = 'x"]


@pytest.mark.parametrize("sql", UNTOKENIZABLE)
def test_check_select_rejects_sql_that_cannot_be_tokenized(sql):
    result = check_select(sql)
    assert not result.ok
    assert "could not be parsed" in result.reason


@pytest.mark.parametrize("sql", UNTOKENIZABLE)
def test_referenced_tables_of_untokenizable_sql_is_empty(sql):
    assert referenced_tables(sql) == set()
