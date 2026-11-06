"""Small OpenAI-compatible chat-completions client for the pipeline."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from schemagraph.config import LLMConfig, get_config


class LLMError(RuntimeError):
    """The configured chat service failed or returned an unusable response."""


def chat_completion(messages: list[dict[str, str]]) -> tuple[str, int, int]:
    """Return assistant text and prompt/completion token counts."""
    config = get_config().llm
    if config is None:
        raise RuntimeError("LLM configuration is required; add an `llm` section to config.yml")
    return complete(config, messages)


def complete(config: LLMConfig, messages: list[dict[str, str]]) -> tuple[str, int, int]:
    """Send one non-streaming request to an OpenAI-compatible chat endpoint."""
    headers = {"Content-Type": "application/json"}
    key = config.api_key
    if key is not None:
        headers["Authorization"] = f"Bearer {key.get_secret_value()}"

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError(f"chat service returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LLMError(f"could not reach chat service: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LLMError("chat service returned invalid JSON") from exc

    try:
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError("chat service response is missing message content or has invalid usage data") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError("chat service returned empty assistant content")
    return content.strip(), input_tokens, output_tokens
