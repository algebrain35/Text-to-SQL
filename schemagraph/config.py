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
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
}
PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,30}$")


class EmbeddingConfig(BaseModel):
    """Which embedding model to use for retrieval, and how to reach it.

    provider
        `openai`: OpenAI, or any server with an OpenAI-style /embeddings endpoint
        (Ollama's /v1, vLLM, ...). `voyage`: Voyage AI. Anthropic has no embedding model
        of its own and recommends Voyage, so `anthropic` is accepted as another name
        for `voyage`. `gemini`: Google's Gemini API (`gemini-embedding-2`, `gemini-embedding-001`).
    name
        Names this set of vectors in Neo4j. The default keeps the plain index and property
        names; any other name adds a suffix, so vectors from different models can sit
        side by side in one graph and be compared (see graph/vector.py).

    max_wait_s / requests_per_minute
        Rate limits (HTTP 429, common on free tiers): a limited request waits and retries
        (honouring Retry-After) for at most `max_wait_s` seconds in total, then gives up with
        a "rate limited" result and the embeddings stored so far are kept. `requests_per_minute`
        spaces requests out so the limit is not hit in the first place.

    Defaults for base_url and api_key_env depend on the provider. Set `api_key_env: null`
    for a local server that needs no key.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "voyage", "gemini", "anthropic"] = "openai"
    name: str = "default"
    base_url: str | None = None            # default: the provider's public endpoint
    model: str
    dimensions: int = Field(ge=1)          # must match the vector index; changing it means re-embedding
    api_key_env: str | None = None         # default: OPENAI_API_KEY / VOYAGE_API_KEY; null means no key
    send_dimensions: bool = False          # ask the API for shortened vectors (OpenAI v3, Voyage 4 support it)
    batch_size: int = Field(default=64, ge=1)   # texts per request (gemini: 100 unless set)
    # Add each table's foreign-key neighbours to its embedded text ("References: ... Referenced by: ..."),
    # so link tables (product, person) match questions about what they link. Changing it changes the text,
    # so embed under a new `name` (or the stored vectors are recomputed) and compare with retrieval_eval.
    include_relationships: bool = False
    timeout_s: float = Field(default=30.0, gt=0)
    max_wait_s: float = Field(default=120.0, ge=0)             # longest total wait on one rate-limited request
    requests_per_minute: int | None = Field(default=None, ge=1)  # pace calls to stay under a free-tier limit
    # Remember each question's embedding in this SQLite file, so a question is paid for once however many modes,
    # settings and reruns ask it (an evaluation sweep embeds the same question dozens of times). Keyed by
    # provider, model, dimensions and text, so changing the model never returns another model's vector.
    # null (the default) turns it off; delete the file to start over.
    query_cache: str | None = None

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
        if self.provider == "gemini" and "batch_size" not in self.model_fields_set:
            self.batch_size = 100  # Gemini accepts up to 100 texts per batchEmbedContents call
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


LLM_PROVIDERS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _provider_defaults(self) -> "LLMConfig":
        if "api_key_env" not in self.model_fields_set:
            self.api_key_env = LLM_PROVIDERS[self.provider]
        return self

    @property
    def api_key(self) -> SecretStr | None:
        if self.api_key_env is None:
            return None
        value = os.environ.get(self.api_key_env)
        if value is None:
            raise RuntimeError(f"Environment variable {self.api_key_env} is not set")
        return SecretStr(value)


class PipelineConfig(BaseModel):
    """Defaults for /api/ask when a request's `options` leave a value out"""
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    row_limit: int = Field(default=200, ge=1)
    fewshot_k: int = Field(default=3, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postgres: PostgresConfig
    # Optional so that extraction-only setups need no Neo4j section. Code that
    # needs the graph calls graph.driver.get_driver(), which says what is missing.
    neo4j: Neo4jConfig | None = None
    embedding: EmbeddingConfig | None = None
    # Optional so extraction and graph building need no LLM, the pipeline checks for it.
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