"""Hindsight-backed memory of CI failures and their outcomes, plus an in-memory fake."""
from __future__ import annotations

import functools
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Protocol

from .models import CIRun, FailureSignature, Memory, Outcome

MENTAL_MODEL_ID = "flaky-ledger"
MENTAL_MODEL_QUERY = (
    "Which CI tests are flaky, how often do they fail and pass on rerun, which failures were "
    "caused by dependencies or infrastructure, and which code fixes resolved recurring failures?"
)
DIRECTIVE = (
    "Never call a failure flaky if a code change fixed the same failure signature before; "
    "call it a regression instead."
)
_SECRET = re.compile(r"(hsk|gsk)_[A-Za-z0-9_\-]+")
BANK_MISSION = (
    "Remember CI failures, reruns, fixes and developer verdicts for this repository so that "
    "recurring and flaky failures are recognised instantly."
)


class MemoryUnavailable(RuntimeError):
    """Hindsight could not be reached or rejected the request."""


class MemoryStore(Protocol):
    def ensure_bank(self) -> None: ...
    def reset(self) -> None: ...
    def record_failure(self, run: CIRun, sig: FailureSignature) -> None: ...
    def record_outcome(self, run: CIRun, sig: FailureSignature, outcome: Outcome) -> None: ...
    def history(self, sig: FailureSignature, limit: int = 12) -> list[Memory]: ...
    def learned_summary(self) -> str: ...
    def refresh_summary(self) -> None: ...
    def ask(self, question: str) -> str: ...


def signature_tags(sig: FailureSignature) -> list[str]:
    tags = [f"sig:{sig.sig_hash}"]
    if sig.test_id:
        tags.append(f"test:{sig.test_id}")
    return tags


def failure_text(run: CIRun, sig: FailureSignature) -> str:
    test = sig.test_id or "none (job-level failure)"
    return (
        f"CI run {run.run_id} on {run.started_at[:10]} failed on branch {run.branch} "
        f"(runner {run.runner}, commit {run.commit[:7]}). Failing test: {test}. "
        f"Error: {sig.error_type}: {sig.message[:300]}. Failure signature {sig.sig_hash}."
    )


def outcome_text(run: CIRun, sig: FailureSignature, outcome: Outcome) -> str:
    test = sig.test_id or "the job"
    rerun = "a rerun passed with no code change" if outcome.rerun_passed else "a rerun failed again"
    note = outcome.fix_note.strip() or "no note"
    return (
        f"Resolution of CI run {run.run_id} ({test}, signature {sig.sig_hash}): {rerun}. "
        f"Developer verdict: {outcome.label}. Fix or note: {note}."
    )


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _describe(exc: Exception) -> str:
    status = _status(exc)
    if status == 401:
        return "Hindsight rejected the API key (401). Check HINDSIGHT_API_KEY in .env."
    if status == 402:
        return "Hindsight Cloud is out of credits (402). Apply the promo code under Billing."
    if status is not None:
        return f"Hindsight request failed with HTTP {status}."
    text = _SECRET.sub("[redacted]", str(exc))[:200]
    return f"Hindsight unreachable ({type(exc).__name__}: {text})"


def _to_memory(result: Any, exact: bool) -> Memory:
    date = (getattr(result, "occurred_start", None) or getattr(result, "mentioned_at", None) or "")[:10]
    metadata = getattr(result, "metadata", None) or {}
    return Memory(
        id=result.id,
        text=result.text,
        date=date or None,
        tags=tuple(getattr(result, "tags", None) or ()),
        run_id=metadata.get("run_id") or getattr(result, "document_id", None),
        exact=exact,
    )


class HindsightStore:
    def __init__(self, client: Any, bank_id: str, repo: str):
        self._client = client
        self.bank_id = bank_id
        self.repo = repo

    @classmethod
    def from_config(cls, cfg: Any, bank_id: str | None = None) -> "HindsightStore":
        from hindsight_client import Hindsight

        client = Hindsight(base_url=cfg.hindsight_url, api_key=cfg.hindsight_api_key, timeout=60.0)
        return cls(client, bank_id or cfg.bank_id, cfg.repo)

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # the client raises ApiException subclasses and transport errors
            raise MemoryUnavailable(_describe(exc)) from exc

    def ensure_bank(self) -> None:
        self._call(
            self._client.create_bank,
            bank_id=self.bank_id,
            name=f"DejaFail {self.repo}",
            mission=BANK_MISSION,
            disposition_skepticism=4,
        )
        try:
            self._client.get_mental_model(self.bank_id, MENTAL_MODEL_ID)
            return
        except Exception as exc:
            if _status(exc) != 404:
                raise MemoryUnavailable(_describe(exc)) from exc
        self._call(
            self._client.create_directive,
            bank_id=self.bank_id,
            name="No flaky verdict after a code fix",
            content=DIRECTIVE,
        )
        self._call(
            self._client.create_mental_model,
            bank_id=self.bank_id,
            name="Flaky ledger",
            source_query=MENTAL_MODEL_QUERY,
            id=MENTAL_MODEL_ID,
        )

    def reset(self) -> None:
        try:
            self._client.delete_bank(self.bank_id)
        except Exception as exc:
            if _status(exc) != 404:
                raise MemoryUnavailable(_describe(exc)) from exc
        self.ensure_bank()

    def record_failure(self, run: CIRun, sig: FailureSignature) -> None:
        self._call(
            self._client.retain,
            bank_id=self.bank_id,
            content=failure_text(run, sig),
            timestamp=_parse_ts(run.started_at),
            context="ci failure",
            document_id=run.run_id,
            metadata={"run_id": run.run_id, "kind": "failure", "branch": run.branch, "runner": run.runner},
            tags=[f"repo:{self.repo}", "kind:failure", *signature_tags(sig)],
            retain_async=False,
        )

    def record_outcome(self, run: CIRun, sig: FailureSignature, outcome: Outcome) -> None:
        self._call(
            self._client.retain,
            bank_id=self.bank_id,
            content=outcome_text(run, sig, outcome),
            timestamp=_parse_ts(run.started_at),
            context="ci failure resolution",
            document_id=f"{run.run_id}-outcome",
            metadata={
                "run_id": run.run_id,
                "kind": "outcome",
                "label": outcome.label,
                "rerun_passed": str(outcome.rerun_passed).lower(),
            },
            tags=[f"repo:{self.repo}", "kind:outcome", f"label:{outcome.label}", *signature_tags(sig)],
            retain_async=False,
        )

    def history(self, sig: FailureSignature, limit: int = 12) -> list[Memory]:
        about = f"{sig.error_type}: {sig.message[:200]}"
        tagged = self._call(
            self._client.recall,
            bank_id=self.bank_id,
            query=f"{sig.test_id or 'CI job'} failed with {about}",
            tags=signature_tags(sig),
            tags_match="any_strict",
            budget="mid",
            max_tokens=2048,
        )
        similar = self._call(
            self._client.recall, bank_id=self.bank_id, query=about, budget="low", max_tokens=1024
        )
        seen: set[str] = set()
        out: list[Memory] = []
        batches = [(tagged.results or [], True), (similar.results or [], False)]
        for results, exact in batches:
            for result in results:
                if result.id in seen:
                    continue
                seen.add(result.id)
                out.append(_to_memory(result, exact))
                if len(out) >= limit:
                    return out
        return out

    def learned_summary(self) -> str:
        try:
            model = self._client.get_mental_model(self.bank_id, MENTAL_MODEL_ID, detail="content")
        except Exception as exc:
            if _status(exc) == 404:
                return ""
            raise MemoryUnavailable(_describe(exc)) from exc
        return getattr(model, "content", None) or ""

    def refresh_summary(self) -> None:
        self._call(self._client.refresh_mental_model, self.bank_id, MENTAL_MODEL_ID)

    def ask(self, question: str) -> str:
        response = self._call(self._client.reflect, bank_id=self.bank_id, query=question, budget="mid")
        return response.text or ""


class ThreadConfinedStore:
    """Runs every call of a wrapped MemoryStore on one dedicated worker thread.

    The Hindsight client runs its async HTTP calls on the calling thread's event loop and keeps one
    aiohttp session bound to the first loop it used. Streamlit runs each browser session on its own
    thread with its own loop, so a shared client breaks in every session after the first ("Timeout
    context manager should be used inside a task"). Build the store on the worker thread with
    ``build`` so the client and its session live there, and every call goes through that thread.
    Results are returned and exceptions re-raised in the caller; plain attributes pass through.
    """

    def __init__(self, store: Any, executor: ThreadPoolExecutor | None = None):
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="hindsight")
        self._store = store

    @classmethod
    def build(cls, factory: Callable[..., Any], *args: Any, **kwargs: Any) -> "ThreadConfinedStore":
        """Construct the store on the worker thread itself, then wrap it."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hindsight")
        try:
            store = executor.submit(factory, *args, **kwargs).result()
        except BaseException:
            executor.shutdown(wait=False)
            raise
        return cls(store, executor)

    def shutdown(self) -> None:
        """Stop the worker thread once pending calls finish. The store is unusable afterwards."""
        self._executor.shutdown(wait=False)

    def __getattr__(self, name: str) -> Any:
        store = self.__dict__.get("_store")
        if store is None:
            raise AttributeError(name)
        attr = getattr(store, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def on_worker(*args: Any, **kwargs: Any) -> Any:
            return self._executor.submit(attr, *args, **kwargs).result()

        return on_worker


class FakeStore:
    """In-memory stand-in for tests and offline development. Not used in the live demo."""

    def __init__(self) -> None:
        self.records: list[Memory] = []
        self.resets = 0

    def ensure_bank(self) -> None:
        return None

    def reset(self) -> None:
        self.records.clear()
        self.resets += 1

    def record_failure(self, run: CIRun, sig: FailureSignature) -> None:
        tags = ("kind:failure", *signature_tags(sig))
        self.records.append(Memory(f"{run.run_id}-f", failure_text(run, sig), run.started_at[:10], tags, run.run_id))

    def record_outcome(self, run: CIRun, sig: FailureSignature, outcome: Outcome) -> None:
        tags = ("kind:outcome", f"label:{outcome.label}", *signature_tags(sig))
        self.records.append(
            Memory(f"{run.run_id}-o", outcome_text(run, sig, outcome), run.started_at[:10], tags, run.run_id)
        )

    def history(self, sig: FailureSignature, limit: int = 12) -> list[Memory]:
        keys = set(signature_tags(sig))
        hits = [
            Memory(m.id, m.text, m.date, m.tags, m.run_id, exact=True)
            for m in self.records
            if keys & set(m.tags)
        ]
        return list(reversed(hits))[:limit]

    def learned_summary(self) -> str:
        labels = sorted({t for m in self.records for t in m.tags if t.startswith("label:")})
        return f"{len(self.records)} memories; labels seen: {', '.join(labels) or 'none'}"

    def refresh_summary(self) -> None:
        return None

    def ask(self, question: str) -> str:
        return f"(offline) {len(self.records)} memories available for: {question}"
