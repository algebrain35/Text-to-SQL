import io
import json
import urllib.error

import pytest

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


def _http_error(code, body: bytes = b"{}", headers=None):
    return urllib.error.HTTPError("https://example.test/v1/chat/completions", code, "err", headers or {}, io.BytesIO(body))


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (json.dumps({"error": {"message": "The model `x` does not exist", "type": "invalid_request_error"}}).encode(),
         "The model `x` does not exist"),
        (json.dumps([{"error": {"code": 404, "message": "models/gemini-9 is not found", "status": "NOT_FOUND"}}]).encode(),
         "models/gemini-9 is not found"),
        (b"<html>Not Found</html>", "<html>Not Found</html>"),
        (json.dumps({"error": "plain string error"}).encode(), "plain string error"),
        (b"", ""),
    ],
)
def test_error_detail_reads_the_services_explanation(body, expected):
    assert llm.error_detail(body) == expected


def test_error_detail_is_clipped():
    assert len(llm.error_detail(b"x" * 5000)) == 300


def test_http_errors_say_why_and_where(monkeypatch):
    config = LLMConfig(model="gemini-9", base_url="https://example.test/v1", api_key_env=None)
    body = json.dumps([{"error": {"message": "models/gemini-9 is not found"}}]).encode()

    def fake_urlopen(request, timeout):
        raise _http_error(404, body)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(llm.LLMError) as caught:
        llm.complete(config, [{"role": "user", "content": "q"}])
    message = str(caught.value)
    assert message.startswith("chat service returned HTTP 404: models/gemini-9 is not found")
    assert "POST https://example.test/v1/chat/completions" in message and "model 'gemini-9'" in message


def test_http_error_with_an_empty_body_still_names_the_request(monkeypatch):
    config = LLMConfig(model="m", base_url="https://example.test/v1", api_key_env=None, max_retries=0)
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(500, b"")))
    with pytest.raises(llm.LLMError, match=r"HTTP 500 \(POST https://example.test/v1/chat/completions"):
        llm.complete(config, [{"role": "user", "content": "q"}])


def test_the_api_key_never_appears_in_an_error(monkeypatch):
    config = LLMConfig(model="m", base_url="https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(401, b'{"error": {"message": "bad key"}}')))
    with pytest.raises(llm.LLMError) as caught:
        llm.complete(config, [{"role": "user", "content": "q"}])
    assert "sk-very-secret" not in str(caught.value) and "bad key" in str(caught.value)


# --- retries and request shape ------------------------------------------------


def _script(monkeypatch, outcomes):
    """urlopen returns or raises each outcome in turn; returns the list of sleeps and the call count."""
    sleeps, calls = [], []

    def fake_urlopen(request, timeout):
        calls.append(request)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    return sleeps, calls


MESSAGES = [{"role": "user", "content": "q"}]


def test_a_429_is_retried_and_waits_what_the_service_asked(monkeypatch):
    sleeps, calls = _script(monkeypatch, [_http_error(429, headers={"Retry-After": "7"}), _Response()])
    result = llm.complete(LLMConfig(model="m", api_key_env=None), MESSAGES)
    assert result[0] == "SELECT 1" and len(calls) == 2
    assert sleeps == [7.0]


def test_backoff_doubles_and_gives_up_after_max_retries(monkeypatch):
    sleeps, calls = _script(monkeypatch, [_http_error(503)] * 4)
    with pytest.raises(llm.LLMError, match="HTTP 503"):
        llm.complete(LLMConfig(model="m", api_key_env=None, max_retries=3), MESSAGES)
    assert len(calls) == 4 and sleeps == [2.0, 4.0, 8.0]


def test_a_wait_longer_than_max_wait_fails_at_once_and_says_so(monkeypatch):
    sleeps, calls = _script(monkeypatch, [_http_error(429, headers={"Retry-After": "600"})])
    with pytest.raises(llm.LLMError, match=r"asked for a 600s wait \(max_wait_s is 30\)"):
        llm.complete(LLMConfig(model="m", api_key_env=None), MESSAGES)
    assert len(calls) == 1 and sleeps == []


def test_client_errors_are_not_retried(monkeypatch):
    sleeps, calls = _script(monkeypatch, [_http_error(400, body=b'{"error": {"message": "bad param"}}')])
    with pytest.raises(llm.LLMError, match="HTTP 400: bad param"):
        llm.complete(LLMConfig(model="m", api_key_env=None), MESSAGES)
    assert len(calls) == 1 and sleeps == []


def test_timeouts_and_connection_errors_are_retried(monkeypatch):
    sleeps, _calls = _script(monkeypatch, [TimeoutError("slow"), urllib.error.URLError("reset"), _Response()])
    assert llm.complete(LLMConfig(model="m", api_key_env=None), MESSAGES)[0] == "SELECT 1"
    assert sleeps == [2.0, 4.0]


def test_max_retries_zero_means_one_attempt(monkeypatch):
    _sleeps, calls = _script(monkeypatch, [_http_error(429)])
    with pytest.raises(llm.LLMError, match="HTTP 429"):
        llm.complete(LLMConfig(model="m", api_key_env=None, max_retries=0), MESSAGES)
    assert len(calls) == 1


def test_reasoning_model_settings_change_the_request_body(monkeypatch):
    _, calls = _script(monkeypatch, [_Response()])
    config = LLMConfig(model="m", api_key_env=None, temperature=None, max_tokens_param="max_completion_tokens", max_tokens=900)
    llm.complete(config, MESSAGES)
    body = json.loads(calls[0].data)
    assert body["max_completion_tokens"] == 900
    assert "max_tokens" not in body and "temperature" not in body


def test_default_request_body_is_unchanged(monkeypatch):
    _, calls = _script(monkeypatch, [_Response()])
    llm.complete(LLMConfig(model="m", api_key_env=None), MESSAGES)
    body = json.loads(calls[0].data)
    assert body["max_tokens"] == 2048 and body["temperature"] == 0.0


def test_anthropic_through_its_openai_compatible_endpoint(monkeypatch):
    """What config.yml's anthropic block sends: Bearer key, /v1/chat/completions, and no temperature
    (Claude 4.7 and later reject any value below 1)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    _, calls = _script(monkeypatch, [_Response()])
    config = LLMConfig(base_url="https://api.anthropic.com/v1/", model="claude-sonnet-5-5",
                       api_key_env="ANTHROPIC_API_KEY", temperature=None, max_tokens=4096)
    assert llm.complete(config, MESSAGES)[0] == "SELECT 1"
    request = calls[0]
    assert request.full_url == "https://api.anthropic.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-ant-test"
    body = json.loads(request.data)
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 4096 and "temperature" not in body
