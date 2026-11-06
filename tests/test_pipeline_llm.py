import json

from schemagraph.config import LLMConfig
from schemagraph.pipeline import llm


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        ).encode()


def test_complete_sends_configured_bearer_key_and_parses_usage(monkeypatch):
    config = LLMConfig(model="test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    result = llm.complete(config, [{"role": "user", "content": "count rows"}])

    assert captured["request"].full_url == "https://api.openai.com/v1/chat/completions"
    assert captured["request"].get_header("Authorization") == "Bearer test-secret"
    assert captured["timeout"] == 60.0
    assert result == ("SELECT 1", 4, 2)


def test_complete_omits_authorization_for_local_endpoint(monkeypatch):
    config = LLMConfig(model="test-model", base_url="http://localhost:8000/v1", api_key_env=None)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return _Response()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    llm.complete(config, [{"role": "user", "content": "count rows"}])

    assert captured["request"].get_header("Authorization") is None
