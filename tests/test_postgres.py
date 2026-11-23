"""Unit tests for the pure helpers in db/postgres.py. No database needed."""

from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from schemagraph.config import PostgresConfig, PostgresRole
from schemagraph import db as db_api
from schemagraph.db import postgres
from schemagraph.db.postgres import _strip_trailing_semicolon, _to_str


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (1, "1"),
        (0, "0"),
        (3.5, "3.5"),
        (Decimal("8248.9900"), "8248.9900"),  # trailing zeros preserved
        ("Jon", "Jon"),
        ("", ""),  # empty string is NOT null
        (True, "true"),  # not "True": matches PostgreSQL's own text form
        (False, "false"),
        (date(2024, 1, 31), "2024-01-31"),
        (datetime(2024, 1, 31, 8, 30), "2024-01-31 08:30:00"),
        (b"\x00\xff", "\\x00ff"),
        (bytearray(b"\x01"), "\\x01"),
        (memoryview(b"\x0a"), "\\x0a"),
    ],
)
def test_to_str(value, expected):
    assert _to_str(value) == expected


def test_to_str_never_returns_non_string():
    for value in (1, 2.0, Decimal("1"), True, date.today(), b"x"):
        assert isinstance(_to_str(value), str)


def test_database_package_exposes_only_the_stable_public_api():
    assert db_api.__all__ == [
        "list_databases",
        "run_readonly_query",
        "explain",
        "run_admin_query",
        "is_service_failure",
    ]
    assert all(callable(getattr(db_api, name)) for name in db_api.__all__)


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT 1", "SELECT 1"),
        ("SELECT 1;", "SELECT 1"),
        ("SELECT 1 ;  ", "SELECT 1"),
        ("  SELECT 1;\n", "  SELECT 1"),
        ("SELECT 1;;", "SELECT 1"),
        ("SELECT ';' AS semi", "SELECT ';' AS semi"),  # inner ';' untouched
        ("SELECT 'a;'", "SELECT 'a;'"),  # ';' inside a string at the end is kept
        ("", ""),
    ],
)
def test_strip_trailing_semicolon(sql, expected):
    assert _strip_trailing_semicolon(sql) == expected


def test_list_databases_uses_readonly_role_and_filters_names(monkeypatch):
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, query):
            assert "NOT datistemplate" in query
            assert "datname <> 'postgres'" in query
            assert "has_database_privilege" in query
            assert "ORDER BY datname" in query
            return SimpleNamespace(
                fetchall=lambda: [
                    ("adventureworks",),
                    ("bad-name",),
                    ("bird_staging",),
                ]
            )

    def fake_connect(role, db_id, **kwargs):
        assert role == "readonly"
        assert db_id == "postgres"
        assert kwargs == {}
        return FakeConnection()

    monkeypatch.setattr(postgres, "_connect", fake_connect)

    assert postgres.list_databases() == ["adventureworks", "bird_staging"]


def test_run_readonly_query_uses_readonly_connection_and_caps_rows(monkeypatch):
    class FakeCursor:
        description = [SimpleNamespace(name="id")]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, query):
            assert query == "SELECT id"

        def fetchmany(self, size):
            assert size == 3
            return [(1,), (2,), (3,)]

    class FakeConnection:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self, name):
            assert name == "sg_readonly_cursor"
            return FakeCursor()

    connection = FakeConnection()

    def fake_connect(role, db_id, **kwargs):
        assert role == "readonly"
        assert db_id == "adventureworks"
        return connection

    monkeypatch.setattr(postgres, "_connect", fake_connect)

    result = postgres.run_readonly_query("adventureworks", "SELECT id;", row_limit=2)

    assert connection.read_only is True
    assert result.columns == ["id"]
    assert result.rows == [["1"], ["2"]]
    assert result.row_count == 2
    assert result.truncated is True


def test_run_readonly_query_serializes_supported_result_values(monkeypatch):
    values = (
        datetime(2024, 1, 31, 8, 30),
        Decimal("8248.9900"),
        True,
        b"\x00\xff",
        None,
    )

    class FakeCursor:
        description = [
            SimpleNamespace(name=name)
            for name in ("created_at", "amount", "active", "payload", "missing")
        ]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, query):
            assert query == "SELECT values"

        def fetchmany(self, size):
            assert size == 2
            return [values]

    class FakeConnection:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self, name):
            assert name == "sg_readonly_cursor"
            return FakeCursor()

    monkeypatch.setattr(
        postgres,
        "_connect",
        lambda role, db_id, **kwargs: FakeConnection(),
    )

    result = postgres.run_readonly_query("adventureworks", "SELECT values;", row_limit=1)

    assert result.columns == ["created_at", "amount", "active", "payload", "missing"]
    assert result.rows == [
        ["2024-01-31 08:30:00", "8248.9900", "true", "\\x00ff", None]
    ]
    assert result.row_count == 1
    assert result.truncated is False


def test_readonly_connection_uses_readonly_role_and_server_guardrails(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-secret")
    monkeypatch.setenv("READONLY_PASSWORD", "readonly-secret")
    config = PostgresConfig(
        host="localhost",
        port=5432,
        statement_timeout_ms=2500,
        admin=PostgresRole(user="sg_admin", password_env="ADMIN_PASSWORD"),
        readonly=PostgresRole(user="sg_readonly", password_env="READONLY_PASSWORD"),
    )

    conninfo = conninfo_to_dict(config.conninfo("readonly", "adventureworks"))

    assert conninfo["user"] == "sg_readonly"
    assert conninfo["password"] == "readonly-secret"
    assert "-c statement_timeout=2500" in conninfo["options"]
    assert "-c default_transaction_read_only=on" in conninfo["options"]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE customers SET name = 'x'",
        "SELECT * FROM customers; DELETE FROM customers",
        "SELECT * INTO copied_customers FROM customers",
        "WITH changed AS (DELETE FROM customers RETURNING *) SELECT * FROM changed",
        "SELECT * FROM customers FOR UPDATE",
    ],
)
def test_run_readonly_query_rejects_unsafe_sql_before_connecting(monkeypatch, statement):
    def unexpected_connect(*args, **kwargs):
        pytest.fail("Unsafe SQL must be rejected before connecting to PostgreSQL")

    monkeypatch.setattr(postgres, "_connect", unexpected_connect)

    with pytest.raises(ValueError, match="single read-only SELECT"):
        postgres.run_readonly_query("adventureworks", statement, row_limit=10)


def test_explain_rejects_unsafe_sql_before_connecting(monkeypatch):
    def unexpected_connect(*args, **kwargs):
        pytest.fail("Unsafe SQL must be rejected before connecting to PostgreSQL")

    monkeypatch.setattr(postgres, "_connect", unexpected_connect)

    result = postgres.explain("adventureworks", "DELETE FROM customers")

    assert result.ok is False
    assert result.error == "Only a single read-only SELECT statement is allowed."


@pytest.mark.parametrize("exc, expected", [
    (psycopg.OperationalError("connection refused"), True),       # no answer from the server: no SQLSTATE
    (psycopg.errors.ConnectionFailure("lost"), True),              # 08006
    (psycopg.errors.AdminShutdown("shutting down"), True),         # 57P01
    (psycopg.errors.TooManyConnections("full"), True),             # 53300
    (psycopg.errors.QueryCanceled("statement timeout"), False),    # 57014: the query ran too long
    (psycopg.errors.UndefinedColumn("no such column"), False),
    (psycopg.errors.SyntaxError("bad sql"), False),
    (psycopg.DataError("could not decode"), False),                # client-side, but not a connection problem
    (ValueError("not psycopg"), False),
])
def test_is_service_failure_separates_outages_from_query_errors(exc, expected):
    assert postgres.is_service_failure(exc) is expected
