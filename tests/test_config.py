"""Unit tests for config.py: the two-role connection rules. No database needed."""

import pytest
from pydantic import ValidationError

from schemagraph.config import AppConfig, load_config

CONFIG_YAML = """
postgres:
  host: dbhost
  port: 5433
  sslmode: prefer
  statement_timeout_ms: 7000
  admin:    {user: sg_admin,    password_env: TEST_ADMIN_PW}
  readonly: {user: sg_readonly, password_env: TEST_READONLY_PW}
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("TEST_ADMIN_PW", "adminpw")
    monkeypatch.setenv("TEST_READONLY_PW", "readpw")
    path = tmp_path / "config.yml"
    path.write_text(CONFIG_YAML)
    return load_config(path)


def test_readonly_conninfo_sets_timeout_and_read_only(cfg):
    info = cfg.postgres.conninfo("readonly", "adventureworks")
    assert "user=sg_readonly" in info
    assert "dbname=adventureworks" in info
    assert "statement_timeout=7000" in info
    assert "default_transaction_read_only=on" in info


def test_admin_conninfo_has_no_read_only_options(cfg):
    info = cfg.postgres.conninfo("admin", "adventureworks")
    assert "user=sg_admin" in info
    assert "statement_timeout" not in info
    assert "read_only" not in info


@pytest.mark.parametrize("role", ["admin", "readonly"])
def test_client_encoding_is_utf8(cfg, role):
    # SQL_ASCII databases otherwise return bytes instead of str.
    assert "client_encoding=utf8" in cfg.postgres.conninfo(role, "x")


def test_roles_use_their_own_password(cfg):
    assert "password=adminpw" in cfg.postgres.conninfo("admin", "x")
    assert "password=readpw" in cfg.postgres.conninfo("readonly", "x")


@pytest.mark.parametrize("bad", ["", "Adventure", "a b", "a;b", "a-b", "x' OR 1=1", "a/b"])
def test_invalid_db_id_rejected(cfg, bad):
    with pytest.raises(ValueError):
        cfg.postgres.conninfo("readonly", bad)


def test_missing_password_env_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_ADMIN_PW", raising=False)
    monkeypatch.setenv("TEST_READONLY_PW", "readpw")
    path = tmp_path / "config.yml"
    path.write_text(CONFIG_YAML)
    cfg = load_config(path)
    with pytest.raises(RuntimeError, match="TEST_ADMIN_PW"):
        cfg.postgres.conninfo("admin", "x")


def test_password_is_not_printed(cfg):
    assert "adminpw" not in repr(cfg.postgres.admin.password)


def test_llm_and_pipeline_settings_load_without_storing_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "not-in-config")
    path = tmp_path / "config.yml"
    path.write_text(
        CONFIG_YAML
        + """
llm:
  model: test-model
  api_key_env: TEST_LLM_KEY
  base_url: http://localhost:1234/v1
pipeline:
  top_k: 4
  max_attempts: 2
  row_limit: 50
  examples_k: 1
"""
    )

    loaded = load_config(path)

    assert loaded.llm is not None
    assert loaded.llm.model == "test-model"
    assert loaded.llm.api_key.get_secret_value() == "not-in-config"
    assert loaded.pipeline.top_k == 4
    assert loaded.pipeline.max_attempts == 2
    assert loaded.pipeline.row_limit == 50
    assert loaded.pipeline.examples_k == 1


def test_pipeline_settings_validate_runtime_limits(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(CONFIG_YAML + "pipeline:\n  max_attempts: 0\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(CONFIG_YAML + "  surprise: true\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yml")
    empty = tmp_path / "empty.yml"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_config(empty)
