"""Validation helpers for SQL accepted from the language model.

Everything here is pure: it looks only at the SQL text, never at a database,
config or the network, so `db/` and `pipeline/` can both import it. It is a
syntactic gate, not the only protection. The `sg_readonly` role (SELECT only),
read-only transactions and `statement_timeout` still apply to whatever passes.
"""

from __future__ import annotations

from typing import NamedTuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# Functions a plain SELECT can call that read files, change session settings,
# take locks, send signals, touch sequences, or run SQL held in a string.
# Matched by name (any schema qualifier is ignored), case-insensitively.
DENIED_FUNCTIONS = frozenset(
    {
        "set_config",
        "nextval",
        "setval",
        "pg_notify",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "pg_rotate_logfile",
        "pg_stat_file",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
    }
)
DENIED_FUNCTION_PREFIXES = (
    "pg_sleep",  # pg_sleep, pg_sleep_for, pg_sleep_until
    "pg_read_",  # pg_read_file, pg_read_binary_file
    "pg_ls_",
    "pg_advisory_",
    "pg_logical_",
    "lo_",  # large objects: lo_import, lo_export, lo_get ...
    "dblink",
    "table_to_xml",
    "schema_to_xml",
    "database_to_xml",
)

_NOT_SELECT = "Only a single read-only SELECT statement is allowed"


class SqlCheck(NamedTuple):
    """Outcome of check_select. `reason` is written so it can be shown to the model."""

    ok: bool
    reason: str | None = None


def _function_name(node: exp.Func) -> str:
    name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
    return name.lower()


def _is_denied(name: str) -> bool:
    return name in DENIED_FUNCTIONS or name.startswith(DENIED_FUNCTION_PREFIXES)


def _unwrap(node: exp.Expression) -> exp.Expression:
    """'(SELECT 1)' parses as a Subquery around the select; look through the parentheses."""
    while isinstance(node, exp.Subquery) and not node.alias:
        node = node.this
    return node


def check_select(statement: object) -> SqlCheck:
    """Check that SQL is exactly one read-only SELECT, and say why if not.

    Parsing uses PostgreSQL syntax. Rejected: empty or unparsable text, more than
    one statement, anything but SELECT (or UNION/INTERSECT/EXCEPT of SELECTs),
    data-modifying CTEs, SELECT INTO, locking clauses (FOR UPDATE), and calls to
    the functions in DENIED_FUNCTIONS.
    """
    if not isinstance(statement, str) or not statement.strip():
        return SqlCheck(False, "The SQL is empty.")

    try:
        statements = [node for node in sqlglot.parse(statement, read="postgres") if node]
    except SqlglotError:  # ParseError, and TokenError for an unterminated quote
        return SqlCheck(False, "The SQL could not be parsed as PostgreSQL.")

    if len(statements) != 1:
        return SqlCheck(False, f"{_NOT_SELECT}: found {len(statements)} statements.")

    query = _unwrap(statements[0])
    if not isinstance(query, (exp.Select, exp.SetOperation)):
        kind = type(query).__name__.upper()
        return SqlCheck(False, f"{_NOT_SELECT}: found {kind}.")

    for node in query.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Merge)):
            kind = type(node).__name__.upper()
            return SqlCheck(False, f"{_NOT_SELECT}: {kind} is not allowed, even inside a WITH clause.")

    for select in query.find_all(exp.Select):
        if select.args.get("into"):
            return SqlCheck(False, f"{_NOT_SELECT}: SELECT ... INTO creates a table.")
        if select.args.get("locks"):
            return SqlCheck(False, f"{_NOT_SELECT}: locking clauses such as FOR UPDATE are not allowed.")

    for func in query.find_all(exp.Func):
        name = _function_name(func)
        if _is_denied(name):
            return SqlCheck(False, f"The function {name}() is not allowed in generated SQL.")

    return SqlCheck(True)


def is_single_select(statement: object) -> bool:
    """Return whether SQL is exactly one read-only SELECT statement."""
    return check_select(statement).ok


def referenced_tables(statement: str) -> set[str]:
    """Tables a query reads, as 'schema.table' ('table' when unqualified), lowercase.

    Names defined by a WITH clause are not tables and are left out, as are table
    functions. Returns an empty set for SQL that does not parse.
    """
    try:
        parsed = [node for node in sqlglot.parse(statement, read="postgres") if node]
    except SqlglotError:
        return set()
    found: set[str] = set()
    for tree in parsed:
        ctes = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
        for table in tree.find_all(exp.Table):
            if not table.name:
                continue
            name, schema = table.name.lower(), table.db.lower()
            if not schema and name in ctes:
                continue
            found.add(f"{schema}.{name}" if schema else name)
    return found
