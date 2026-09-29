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
