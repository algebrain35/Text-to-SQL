# Sends chat messages to an OpenAI-style API and returns the reply.

import json
import urllib.error
import urllib.request

from schemagraph.config import get_config


class LLMError(RuntimeError):
    # Raised when the chat service fails or sends back a bad reply
    pass


def chat_completion(messages):
    # Uses the llm settings from config.yml
    config = get_config().llm
    if config is None:
        raise RuntimeError("LLM configuration is required; add an `llm` section to config.yml")
    return complete(config, messages)


def complete(config, messages):
    # Sends the messages and returns (reply, input_tokens, output_tokens)

    url = config.base_url.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if config.api_key is not None:
        headers["Authorization"] = "Bearer " + config.api_key.get_secret_value()

    data = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    # Send the request and read the reply
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"chat service returned HTTP {e.code}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"could not reach chat service: {e}")
    except ValueError:
        raise LLMError("chat service returned invalid JSON")

    # Pull the reply text and token counts out of the response
    try:
        reply = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
    except (KeyError, IndexError, TypeError, ValueError):
        raise LLMError("chat service response is missing message content or has invalid usage data")

    if not isinstance(reply, str) or reply.strip() == "":
        raise LLMError("chat service returned empty assistant content")

    return reply.strip(), input_tokens, output_tokens
