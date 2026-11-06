"""Application configuration, loaded from YAML.

Secrets never live in the YAML file: each role names an environment variable
(`password_env`) that holds its password.

Other sections (neo4j, llm, pipeline, graph) are added as the modules that
need them are built. Changes to this file need review from everyone affected.
"""

from __future__ import annotations

import os
import re
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal

import yaml
from psycopg.conninfo import make_conninfo
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

DEFAULT_CONFIG_PATH = "config.yml"
CONFIG_PATH_ENV = "SCHEMAGRAPH_CONFIG"

# db_id doubles as the PostgreSQL database name; keep it boring and safe.
DB_ID_RE = re.compile(r"^[a-z0-9_]+$")


class PostgresRole(BaseModel):
    """One PostgreSQL login: a user name plus the env var holding its password."""

    model_config = ConfigDict(extra="forbid")

    user: str
    password_env: str

    @cached_property
    def password(self) -> SecretStr:
        value = os.environ.get(self.password_env)
        if value is None:
            raise RuntimeError(f"Environment variable {self.password_env} is not set")
        return SecretStr(value)


class PostgresConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    sslmode: str = "prefer"
    statement_timeout_ms: int = Field(default=10000, ge=1)

    # Two separate logins (SPEC section 7):
    #   admin    - extraction, inference, data loading. Never runs LLM SQL.
    #   readonly - SELECT only. Runs every LLM-generated query.
    admin: PostgresRole
    readonly: PostgresRole

    def conninfo(self, role: Literal["admin", "readonly"], db_id: str) -> str:
        """Build a libpq connection string for one role and one database.

        Each db_id is its own PostgreSQL database with the same name.
        """
        if not DB_ID_RE.fullmatch(db_id):
            raise ValueError(f"Invalid db_id {db_id!r}: use lowercase letters, digits, underscores")

        creds = self.admin if role == "admin" else self.readonly
        params: dict[str, str | int] = {
            "host": self.host,
            "port": self.port,
            "dbname": db_id,
            "user": creds.user,
            "password": creds.password.get_secret_value(),
            "sslmode": self.sslmode,
            # Without this, a SQL_ASCII database (common for restored sample
            # data) makes psycopg return bytes instead of str for every text.
            "client_encoding": "utf8",
        }
        if role == "readonly":
            # Enforced per connection, so no caller can forget to set them:
            # the timeout from SPEC 7, and read-only transactions as a second
            # lock on top of the role's SELECT-only grants.
            params["options"] = (
                f"-c statement_timeout={self.statement_timeout_ms} "
                "-c default_transaction_read_only=on"
            )
        return make_conninfo(**params)


class Neo4jConfig(BaseModel):
    """Connection to the Neo4j graph store. The password comes from an env var."""

    model_config = ConfigDict(extra="forbid")

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    # Community edition has a single database, named "neo4j". Every db_id lives in
    # it, kept apart by the db_id property on each node.
    database: str = "neo4j"

    @cached_property
    def password(self) -> SecretStr:
        value = os.environ.get(self.password_env)
        if value is None:
            raise RuntimeError(f"Environment variable {self.password_env} is not set")
        return SecretStr(value)


EMBEDDING_PROVIDERS = {
    # provider: (default base_url, default api key variable)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "voyage": ("https://ai.mongodb.com/v1", "VOYAGE_API_KEY"),
}
PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


class EmbeddingConfig(BaseModel):
    """Which embedding model to use for retrieval, and how to reach it.

    provider
        `openai`: OpenAI, or any server with an OpenAI-style /embeddings endpoint
        (Ollama's /v1, vLLM, ...). `voyage`: Voyage AI. Anthropic has no embedding model
        of its own and recommends Voyage, so `anthropic` is accepted as another name
        for `voyage`.
    name
        Names this set of vectors in Neo4j. The default keeps the plain index and property
        names; any other name adds a suffix, so vectors from different models can sit
        side by side in one graph and be compared (see graph/vector.py).

    Defaults for base_url and api_key_env depend on the provider. Set `api_key_env: null`
    for a local server that needs no key.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "voyage", "anthropic"] = "openai"
    name: str = "default"
    base_url: str | None = None            # default: the provider's public endpoint
    model: str
    dimensions: int = Field(ge=1)          # must match the vector index; changing it means re-embedding
    api_key_env: str | None = None         # default: OPENAI_API_KEY / VOYAGE_API_KEY; null means no key
    send_dimensions: bool = False          # ask the API for shortened vectors (OpenAI v3, Voyage 4 support it)
    batch_size: int = Field(default=64, ge=1)
    timeout_s: float = Field(default=30.0, gt=0)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not PROFILE_NAME_RE.match(value):
            raise ValueError("name must be lowercase letters, digits and _, starting with a letter (max 31)")
        return value

    @model_validator(mode="after")
    def _provider_defaults(self) -> "EmbeddingConfig":
        if self.provider == "anthropic":
            self.provider = "voyage"
        base_url, key_env = EMBEDDING_PROVIDERS[self.provider]
        if self.base_url is None:
            self.base_url = base_url
        if "api_key_env" not in self.model_fields_set:  # an explicit null stays null
            self.api_key_env = key_env
        return self

    @property
    def api_key(self) -> SecretStr | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        return SecretStr(value)


class LLMConfig(BaseModel):
    """OpenAI-compatible chat-completions endpoint used by the question pipeline."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    model: str
    api_key_env: str | None = "OPENAI_API_KEY"
    timeout_s: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)

    @cached_property
    def api_key(self) -> SecretStr | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        return SecretStr(value)


class PipelineConfig(BaseModel):
    """Default runtime limits for the question-answering pipeline."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1, le=100)
    max_attempts: int = Field(default=3, ge=1, le=10)
    row_limit: int = Field(default=100, ge=1, le=100_000)
    examples_k: int = Field(default=3, ge=0, le=20)
    fewshot_path: Path = Path("schemagraph/eval/questions/fewshot_examples.jsonl")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postgres: PostgresConfig
    # Optional so that extraction-only setups need no Neo4j section. Code that
    # needs the graph calls graph.driver.get_driver(), which says what is missing.
    neo4j: Neo4jConfig | None = None
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    # graph: GraphConfig         <- max_bridge_tables, weight penalties


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not Path(path).exists():
        raise FileNotFoundError(f"Error: configuration file `{path}` not found.")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Configuration file `{path}` is empty.")
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached config for normal use. Tests can call load_config() directly,
    or get_config.cache_clear() after changing SCHEMAGRAPH_CONFIG."""
    return load_config(os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))