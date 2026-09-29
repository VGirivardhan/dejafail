"""Groq chat completions in JSON mode, with validation-driven retries.

Uses the OpenAI SDK pointed at Groq's OpenAI-compatible endpoint. No function
calling: the model returns one JSON object that we parse and validate ourselves.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol

import openai

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_THINK = re.compile(r"<think>.*?</think>", re.S)

Validator = Callable[[dict[str, Any]], None]


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMOutputError(LLMError):
    """The model never produced valid JSON for the schema."""


class LLMUnavailable(LLMError):
    """Groq could not be reached or kept failing."""


class LLM(Protocol):
    def complete_json(self, system: str, user: str, validate: Validator | None = None) -> dict: ...


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _THINK.sub("", text or "").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in model output") from None
        obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def _is_json_validate_failed(exc: Exception) -> bool:
    return getattr(exc, "code", None) == "json_validate_failed" or "json_validate_failed" in str(exc)


def effective_reasoning_effort(model: str, reasoning_effort: str | None) -> str | None:
    """The reasoning_effort actually sent to Groq: only gpt-oss models accept it, and only when set."""
    if model.startswith("openai/gpt-oss") and reasoning_effort:
        return reasoning_effort
    return None


class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        client: Any = None,
        retries: int = 2,
        reasoning_effort: str = "low",
        max_completion_tokens: int = 1024,
    ):
        self.model = model
        self.retries = retries
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=GROQ_BASE_URL, max_retries=4, timeout=60.0
        )

    def _request_options(self) -> dict[str, Any]:
        # Groq's free tier counts reasoning tokens against 8K tokens/minute and 200K tokens/day per model.
        options: dict[str, Any] = {"max_completion_tokens": self.max_completion_tokens}
        effort = effective_reasoning_effort(self.model, self.reasoning_effort)
        if effort is not None:
            options["reasoning_effort"] = effort
        return options

    def complete_json(self, system: str, user: str, validate: Validator | None = None) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = "no attempt made"
        for _ in range(self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    **self._request_options(),
                )
            except openai.APIStatusError as exc:
                if getattr(exc, "status_code", None) == 400:
                    if _is_json_validate_failed(exc):
                        # Groq rejects malformed JSON-mode output with a 400 (json_validate_failed).
                        last_error = str(exc)
                        continue
                    raise LLMUnavailable(f"Groq rejected the request (400): {exc}") from exc
                raise LLMUnavailable(f"Groq request failed: {exc}") from exc
            except openai.APIError as exc:
                raise LLMUnavailable(f"Groq unreachable: {exc}") from exc
            content = response.choices[0].message.content or ""
            try:
                obj = parse_json_object(content)
                if validate is not None:
                    validate(obj)
                return obj
            except ValueError as exc:
                last_error = str(exc)
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"Your reply was invalid: {exc}. Reply with ONLY one JSON object matching the schema.",
                    },
                ]
        raise LLMOutputError(last_error)