import openai
import pytest

from dejafail.llm import GroqLLM, LLMOutputError, LLMUnavailable, parse_json_object
from tests.fakes import fake_openai_client


def _bare(exc_type, message, **attrs):
    """Build an openai exception without its transport-specific constructor."""
    exc = exc_type.__new__(exc_type)
    Exception.__init__(exc, message)
    exc.message = message
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


def test_parse_plain_fenced_and_think_wrapped_json():
    assert parse_json_object('{"kind": "flaky"}') == {"kind": "flaky"}
    assert parse_json_object('```json\n{"kind": "infra"}\n```') == {"kind": "infra"}
    assert parse_json_object('<think>hmm {not json}</think>\n{"kind": "regression"}') == {"kind": "regression"}


def test_parse_rejects_non_object():
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_retries_after_invalid_json_then_succeeds():
    client = fake_openai_client(["not json at all", '{"kind": "flaky"}'])
    llm = GroqLLM(api_key="unused", client=client)
    assert llm.complete_json("sys", "user") == {"kind": "flaky"}
    requests = client.chat.completions.requests
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "invalid" in requests[1]["messages"][-1]["content"]


def test_validation_failure_exhausts_retries():
    client = fake_openai_client(['{"kind": "nope"}'] * 3)
    llm = GroqLLM(api_key="unused", client=client, retries=2)

    def validate(obj):
        raise ValueError("kind must be one of flaky, regression")

    with pytest.raises(LLMOutputError):
        llm.complete_json("sys", "user", validate=validate)
    assert len(client.chat.completions.requests) == 3


def test_groq_json_validate_failed_400_is_retried():
    bad = _bare(openai.BadRequestError, "json_validate_failed", status_code=400)
    client = fake_openai_client([bad, '{"kind": "infra"}'])
    llm = GroqLLM(api_key="unused", client=client)
    assert llm.complete_json("sys", "user") == {"kind": "infra"}


def test_connection_error_becomes_llm_unavailable():
    down = _bare(openai.APIConnectionError, "Connection error.")
    client = fake_openai_client([down])
    llm = GroqLLM(api_key="unused", client=client)
    with pytest.raises(LLMUnavailable):
        llm.complete_json("sys", "user")


def test_gpt_oss_request_sends_reasoning_effort_and_completion_cap():
    client = fake_openai_client(['{"kind": "flaky"}'])
    llm = GroqLLM(api_key="unused", model="openai/gpt-oss-120b", client=client, reasoning_effort="low")
    llm.complete_json("sys", "user")
    request = client.chat.completions.requests[0]
    assert request["reasoning_effort"] == "low"
    assert request["max_completion_tokens"] == 1024


def test_default_llm_asks_for_low_reasoning_effort():
    client = fake_openai_client(['{"kind": "flaky"}'])
    GroqLLM(api_key="unused", client=client).complete_json("sys", "user")
    assert client.chat.completions.requests[0]["reasoning_effort"] == "low"


def test_other_models_omit_reasoning_effort_but_keep_completion_cap():
    client = fake_openai_client(['{"kind": "flaky"}'])
    llm = GroqLLM(api_key="unused", model="qwen/qwen3-32b", client=client, reasoning_effort="low")
    llm.complete_json("sys", "user")
    request = client.chat.completions.requests[0]
    assert "reasoning_effort" not in request
    assert request["max_completion_tokens"] == 1024


def test_empty_reasoning_effort_is_not_sent():
    client = fake_openai_client(['{"kind": "flaky"}'])
    llm = GroqLLM(api_key="unused", model="openai/gpt-oss-20b", client=client, reasoning_effort="")
    llm.complete_json("sys", "user")
    assert "reasoning_effort" not in client.chat.completions.requests[0]


def test_completion_cap_is_configurable():
    client = fake_openai_client(['{"kind": "flaky"}'])
    GroqLLM(api_key="unused", client=client, max_completion_tokens=512).complete_json("sys", "user")
    assert client.chat.completions.requests[0]["max_completion_tokens"] == 512
