"""Asking the LLM for SQL
    build_prompt   question + schema context + examples (+ earlier failed tries) -> messages
    generate_sql   sends the messages, get back a GenerationResult
    extract_sql    pull the SQL out of the model's reply



The chat model is made by `get_chat_model()` from the `llm:` section of config.yml.
Every function that calls the model takes an optional `llm`, so tests pass a fake.
"""
from __future__ import annotations
import re
from functools import lru_cache
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from schemagraph.config import LLMConfig, get_config
from schemagraph.models import Attempt, FewShotExample, GenerationResult, SchemaContext

TEMPERATURE = 0
SYSTEM_PROMPT = """You write PostgreSQL queries that answer questions about a database.

Rules:
- Write exactly one read-only SELECT statement (a WITH ... SELECT is fine). Never modify data.
- Use only the tables and columns in the schema below. Do not invent names.
- Always write table names with their schema, e.g. sales.customer, and give tables short aliases.
- If join conditions are listed, use those exact conditions to connect the tables.
- Return only the columns the question asks for, with readable names.
- Reply with the SQL inside one ```sql code block and nothing else."""

def make_chat_model(cfg: LLMConfig) -> Any:
    """A LangChain chat model for the configured provider which is imported only when needed."""
    key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {}
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return ChatAnthropic(
            model=cfg.model, temperature=TEMPERATURE, max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s, api_key=key, **kwargs,
        )
    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model, temperature=TEMPERATURE, max_tokens=cfg.max_tokens,
            timeout=cfg.timeout_s, api_key=key or "not-needed", base_url=cfg.base_url,
        )
    raise ValueError(f"Unknown LLM provider {cfg.provider!r}")

@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    """The shared chat model from config.yml which raises if the `llm:` section is missing."""
    cfg = get_config().llm
    if cfg is None:
        raise RuntimeError("config.yml has no `llm:` section, add one to use the pipeline")
    return make_chat_model(cfg)

def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)

def token_counts(message: BaseMessage) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

def _schema_section(context: SchemaContext) -> str:
    parts = ["### Schema", context.ddl.strip() or "(no tables)"]
    if context.join_text:
        parts += ["", "### Join conditions", context.join_text.strip()]
    if context.notes:
        parts += ["", "### Notes"] + [f"- {n}" for n in context.notes]
    return "\n".join(parts)

def _examples_section(examples: list[FewShotExample]) -> str:
    if not examples:
        return ""
    lines = ["### Examples from this database"]
    for ex in examples:
        lines.append(f"Question: {ex.question}")
        if ex.evidence:
            lines.append(f"Hint: {ex.evidence}")
        lines.append(f"```sql\n{ex.gold_sql.strip()}\n```")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(
        question: str,
        context: SchemaContext,
        examples: list[FewShotExample],
        previous_attempts: list[Attempt],
) -> list[BaseMessage]:
    """System rules, then schema, examples, question, then any failed tries."""
    body = [_schema_section(context)]
    examples_text = _examples_section(examples)
    if examples_text:
        body.append(examples_text)
    body.append(f"### Question\n{question.strip()}")
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(body)),
    ]
    for attempt in previous_attempts:
        messages.append(AIMessage(content=f"```sql\n{attempt.sql.strip()}\n```"))
        messages.append(HumanMessage(content=(
            f"That query failed with this error:\n{attempt.error or 'unknown error'}\n\n"
            "Reply with the corrected SQL in one ```sql code block."
        )))
    return messages

_FENCED = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n?(.*?)```", re.DOTALL)
_STARTS_LIKE_SQL = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

def extract_sql(llm_output: str) -> str | None:
    """
    Prefers a ```sql block, then any ``` block, then a reply that is bare SQL.
    """
    blocks = [(lang.lower(), code.strip()) for lang, code in _FENCED.findall(llm_output or "")]
    chosen = next((code for lang, code in blocks if lang == "sql" and code), None)
    if chosen is None:
        chosen = next((code for _, code in blocks if code), None)
    if chosen is None and _STARTS_LIKE_SQL.match(llm_output or ""):
        chosen = llm_output.strip()
    if not chosen:
        return None
    chosen = chosen.strip().rstrip(";").strip()
    return chosen or None

def generate_sql(
        question: str,
        context: SchemaContext,
        examples: list[FewShotExample],
        previous_attempts: list[Attempt],
        *,
        llm: Any | None = None,
) -> GenerationResult:
    """One call to the model. `sql` is None when the reply had no SQL in it."""
    model = llm if llm is not None else get_chat_model()
    reply = model.invoke(build_prompt(question, context, examples, previous_attempts))
    text = message_text(reply)
    input_tokens, output_tokens = token_counts(reply)
    return GenerationResult(
        sql=extract_sql(text), raw_output=text,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
