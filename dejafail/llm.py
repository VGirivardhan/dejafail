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


class GroqLLM:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b", client: Any = None, retries: int = 2):
        self.model = model
        self.retries = retries
        self._client = client or openai.OpenAI(
            api_key=api_key, base_url=GROQ_BASE_URL, max_retries=4, timeout=60.0
        )

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
                )
            except openai.APIStatusError as exc:
                if getattr(exc, "status_code", None) == 400:
                    # Groq rejects malformed JSON-mode output with a 400 (json_validate_failed).
                    last_error = str(exc)
                    continue
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