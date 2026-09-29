"""Network-free stand-ins for Hindsight and Groq used across the test suite."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dejafail.llm import LLMOutputError


class FakeApiError(Exception):
    """Mimics hindsight_client_api ApiException: carries an HTTP status."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def recall_result(id: str, text: str, *, tags=(), run_id=None, date="2026-09-10T09:00:00+00:00"):
    return SimpleNamespace(
        id=id,
        text=text,
        tags=list(tags),
        metadata={"run_id": run_id} if run_id else {},
        document_id=run_id,
        occurred_start=date,
        mentioned_at=None,
    )


class FakeHindsightClient:
    def __init__(self, recall_batches=None, mental_model_exists=True, fail_status=None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.recall_batches = list(recall_batches or [])
        self.mental_model_exists = mental_model_exists
        self.fail_status = fail_status

    def _maybe_fail(self):
        if self.fail_status is not None:
            raise FakeApiError(self.fail_status)

    def create_bank(self, **kwargs):
        self._maybe_fail()
        self.calls.append(("create_bank", kwargs))

    def delete_bank(self, bank_id):
        self.calls.append(("delete_bank", {"bank_id": bank_id}))

    def retain(self, **kwargs):
        self._maybe_fail()
        self.calls.append(("retain", kwargs))

    def recall(self, **kwargs):
        self._maybe_fail()
        self.calls.append(("recall", kwargs))
        batch = self.recall_batches.pop(0) if self.recall_batches else []
        return SimpleNamespace(results=batch)

    def reflect(self, **kwargs):
        self._maybe_fail()
        self.calls.append(("reflect", kwargs))
        return SimpleNamespace(text="Quarantine test_checkout_total.")

    def get_mental_model(self, bank_id, mental_model_id, detail=None):
        self._maybe_fail()
        if not self.mental_model_exists:
            raise FakeApiError(404)
        return SimpleNamespace(content="test_checkout_total is flaky (11 runs).")

    def create_mental_model(self, **kwargs):
        self.calls.append(("create_mental_model", kwargs))
        self.mental_model_exists = True

    def create_directive(self, **kwargs):
        self.calls.append(("create_directive", kwargs))

    def refresh_mental_model(self, bank_id, mental_model_id):
        self.calls.append(("refresh_mental_model", {"id": mental_model_id}))

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class FakeLLM:
    """Scripted LLM. Each reply is a dict (returned) or an Exception (raised)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, validate=None) -> dict:
        self.prompts.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if validate is not None:
            try:
                validate(reply)
            except ValueError as exc:
                raise LLMOutputError(str(exc)) from exc
        return reply


class FakeCompletions:
    """Stands in for OpenAI().chat.completions; each item is a content string or an Exception."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.contents.pop(0)
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_openai_client(contents) -> SimpleNamespace:
    completions = FakeCompletions(contents)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))
