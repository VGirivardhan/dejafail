import asyncio
import threading
from datetime import datetime

import pytest

from dejafail.memory import (
    MENTAL_MODEL_ID,
    FakeStore,
    HindsightStore,
    MemoryUnavailable,
    ThreadConfinedStore,
    failure_text,
    outcome_text,
)
from dejafail.models import CIRun, FailureSignature, Outcome
from dejafail.triage import Triage
from tests.fakes import FakeApiError, FakeHindsightClient, FakeLLM, recall_result

SIG = FailureSignature(
    test_id="tests/test_checkout.py::test_checkout_total",
    error_type="AssertionError",
    message="assert Decimal('0.00') == Decimal('54.97')",
    normalized="assert Decimal('0.00') == Decimal('54.97')",
    sig_hash="abc123def456",
    excerpt="...",
)
RUN = CIRun(
    run_id="run-205",
    started_at="2026-09-09T09:45:00+00:00",
    branch="main",
    runner="gh-ubuntu-x64",
    commit="0123456789abcdef0123456789abcdef01234567",
    log="...",
    truth_label="flaky",
    outcome=Outcome(rerun_passed=True, label="flaky", fix_note="Passed on rerun."),
)


def test_failure_and_outcome_text_are_self_contained():
    text = failure_text(RUN, SIG)
    assert "run-205" in text and "2026-09-09" in text and "test_checkout_total" in text
    assert "abc123def456" in text and "0123456" in text
    out = outcome_text(RUN, SIG, RUN.outcome)
    assert "rerun passed with no code change" in out and "flaky" in out and "Passed on rerun." in out


def test_record_failure_retains_with_tags_metadata_and_sync_flag():
    client = FakeHindsightClient()
    HindsightStore(client, "dejafail-shopfront", "shopfront").record_failure(RUN, SIG)
    name, kwargs = client.calls[-1]
    assert name == "retain"
    assert kwargs["bank_id"] == "dejafail-shopfront"
    assert kwargs["document_id"] == "run-205"
    assert kwargs["retain_async"] is False
    assert isinstance(kwargs["timestamp"], datetime)
    assert {"sig:abc123def456", "test:tests/test_checkout.py::test_checkout_total", "kind:failure"} <= set(kwargs["tags"])
    assert kwargs["metadata"]["run_id"] == "run-205"


def test_record_outcome_uses_separate_document_and_label_tag():
    client = FakeHindsightClient()
    HindsightStore(client, "b", "shopfront").record_outcome(RUN, SIG, RUN.outcome)
    _, kwargs = client.calls[-1]
    assert kwargs["document_id"] == "run-205-outcome"
    assert "label:flaky" in kwargs["tags"]
    assert kwargs["metadata"]["rerun_passed"] == "true"


def test_history_merges_tagged_and_semantic_recall_without_duplicates():
    tagged = [recall_result("f1", "run-201 flaky", tags=["sig:abc123def456"], run_id="run-201")]
    semantic = [
        recall_result("f1", "run-201 flaky", tags=["sig:abc123def456"], run_id="run-201"),
        recall_result("f9", "other failure", tags=["sig:zzz"], run_id="run-230"),
    ]
    client = FakeHindsightClient(recall_batches=[tagged, semantic])
    memories = HindsightStore(client, "b", "shopfront").history(SIG)
    assert [m.id for m in memories] == ["f1", "f9"]
    assert memories[0].exact is True and memories[1].exact is False
    assert memories[0].date == "2026-09-10"
    assert memories[0].run_id == "run-201"
    first_recall = client.calls[0][1]
    assert first_recall["tags_match"] == "any_strict"
    assert "sig:abc123def456" in first_recall["tags"]


def test_history_respects_limit():
    batch = [recall_result(f"f{i}", f"memory {i}") for i in range(20)]
    client = FakeHindsightClient(recall_batches=[batch, []])
    assert len(HindsightStore(client, "b", "shopfront").history(SIG, limit=5)) == 5


def test_tagged_recall_has_4096_token_budget():
    """Test that the tagged history recall uses a 4096-token budget."""
    client = FakeHindsightClient(recall_batches=[[], []])
    HindsightStore(client, "b", "shopfront").history(SIG)
    # First recall call is the tagged one
    first_recall = client.calls[0][1]
    assert first_recall["max_tokens"] == 4096


@pytest.mark.parametrize("status,needle", [(401, "HINDSIGHT_API_KEY"), (402, "credits"), (503, "503")])
def test_api_errors_become_memory_unavailable_with_readable_message(status, needle):
    client = FakeHindsightClient(fail_status=status)
    with pytest.raises(MemoryUnavailable) as exc:
        HindsightStore(client, "b", "shopfront").record_failure(RUN, SIG)
    assert needle in str(exc.value)


def test_ensure_bank_creates_mental_model_and_directive_once():
    client = FakeHindsightClient(mental_model_exists=False)
    store = HindsightStore(client, "b", "shopfront")
    store.ensure_bank()
    store.ensure_bank()
    assert client.names().count("create_bank") == 2
    assert client.names().count("create_mental_model") == 1
    assert client.names().count("create_directive") == 1
    mm = [kw for name, kw in client.calls if name == "create_mental_model"][0]
    assert mm["id"] == MENTAL_MODEL_ID


def test_learned_summary_empty_when_mental_model_missing():
    client = FakeHindsightClient(mental_model_exists=False)
    assert HindsightStore(client, "b", "shopfront").learned_summary() == ""


def test_ask_uses_reflect():
    client = FakeHindsightClient()
    assert "Quarantine" in HindsightStore(client, "b", "shopfront").ask("Which tests should we quarantine?")
    assert client.names() == ["reflect"]


def test_fake_store_history_matches_by_signature_tags():
    store = FakeStore()
    store.record_failure(RUN, SIG)
    store.record_outcome(RUN, SIG, RUN.outcome)
    memories = store.history(SIG)
    assert len(memories) == 2 and all(m.exact for m in memories)
    store.reset()
    assert store.history(SIG) == [] and store.resets == 1


def test_history_with_no_status_raises_memory_unavailable_unreachable():
    """Test that a plain exception (no .status) produces 'unreachable' message."""
    class FakeClientNoStatus(FakeHindsightClient):
        def recall(self, **kwargs):
            raise ConnectionError("Connection refused")

    client = FakeClientNoStatus()
    with pytest.raises(MemoryUnavailable) as exc:
        HindsightStore(client, "b", "shopfront").history(SIG)
    assert "unreachable" in str(exc.value)


def test_error_message_does_not_leak_secret_in_exception():
    """Test that error messages don't leak secret strings from exceptions."""
    class FakeClientWithSecret(FakeHindsightClient):
        def retain(self, **kwargs):
            exc = Exception("hsk_leaky_secret_123 in request")
            exc.status = 401
            raise exc

    client = FakeClientWithSecret()
    with pytest.raises(MemoryUnavailable) as exc:
        HindsightStore(client, "b", "shopfront").record_failure(RUN, SIG)
    error_msg = str(exc.value)
    assert "hsk_leaky_secret_123" not in error_msg
    assert "HINDSIGHT_API_KEY" in error_msg


def test_ensure_bank_retries_after_directive_partial_failure():
    """Test that ensure_bank retries both directive and mental model after partial failure."""
    call_count = {"create_directive": 0}

    class FakeClientPartialFailure(FakeHindsightClient):
        def create_directive(self, **kwargs):
            call_count["create_directive"] += 1
            self.calls.append(("create_directive", kwargs))
            if call_count["create_directive"] == 1:
                raise FakeApiError(503)

    client = FakeClientPartialFailure(mental_model_exists=False)
    store = HindsightStore(client, "b", "shopfront")

    # First call: create_directive fails with 503
    with pytest.raises(MemoryUnavailable) as exc:
        store.ensure_bank()
    assert "503" in str(exc.value)

    # Second call: both create_directive and create_mental_model should be attempted
    store.ensure_bank()

    # Verify create_directive was called twice and create_mental_model once
    assert call_count["create_directive"] == 2
    assert client.names().count("create_directive") == 2
    assert client.names().count("create_mental_model") == 1


def test_status_less_error_keeps_its_cause_and_redacts_keys():
    class FakeClientBrokenLoop(FakeHindsightClient):
        def recall(self, **kwargs):
            raise RuntimeError("Timeout context manager should be used inside a task")

        def retain(self, **kwargs):
            raise OSError("proxy said no to hsk_live-Secret_123 and gsk_abcDEF456")

    store = HindsightStore(FakeClientBrokenLoop(), "b", "shopfront")
    with pytest.raises(MemoryUnavailable) as exc:
        store.history(SIG)
    assert str(exc.value) == (
        "Hindsight unreachable (RuntimeError: Timeout context manager should be used inside a task)"
    )
    with pytest.raises(MemoryUnavailable) as exc:
        store.record_failure(RUN, SIG)
    message = str(exc.value)
    assert message.startswith("Hindsight unreachable (OSError: proxy said no to [redacted] and [redacted]")
    assert "hsk_" not in message and "gsk_" not in message and "Secret_123" not in message


def test_status_less_error_text_is_truncated():
    class FakeClientLongError(FakeHindsightClient):
        def recall(self, **kwargs):
            raise ConnectionError("x" * 500)

    with pytest.raises(MemoryUnavailable) as exc:
        HindsightStore(FakeClientLongError(), "b", "shopfront").history(SIG)
    assert str(exc.value) == f"Hindsight unreachable (ConnectionError: {'x' * 200})"


class ThreadRecordingStore(FakeStore):
    """A FakeStore that notes which thread ran each call, like a client bound to one event loop."""

    def __init__(self) -> None:
        super().__init__()
        self.bank_id = "dejafail-shopfront"
        self.repo = "shopfront"
        self.threads: list[str] = []
        self.built_on = threading.current_thread().name

    def history(self, sig: FailureSignature, limit: int = 12):
        self.threads.append(threading.current_thread().name)
        return super().history(sig, limit)

    def record_failure(self, run: CIRun, sig: FailureSignature) -> None:
        self.threads.append(threading.current_thread().name)
        super().record_failure(run, sig)

    def ask(self, question: str) -> str:
        raise MemoryUnavailable("Hindsight request failed with HTTP 503.")


def test_thread_confined_store_runs_every_call_on_one_worker_thread():
    store = ThreadConfinedStore.build(ThreadRecordingStore)
    try:
        inner = store._store
        assert inner.built_on.startswith("hindsight")
        store.record_failure(RUN, SIG)

        results: list[int] = []

        def caller() -> None:
            loop = asyncio.new_event_loop()  # each Streamlit session thread has its own event loop
            asyncio.set_event_loop(loop)
            try:
                results.append(len(store.history(SIG)))
            finally:
                asyncio.set_event_loop(None)
                loop.close()

        callers = [threading.Thread(target=caller) for _ in range(4)]
        for thread in callers:
            thread.start()
        for thread in callers:
            thread.join()

        assert results == [1, 1, 1, 1]
        assert len(inner.threads) == 5
        assert set(inner.threads) == {inner.built_on}
        assert inner.built_on != threading.current_thread().name
    finally:
        store.shutdown()


def test_thread_confined_store_propagates_exceptions_and_passes_attributes_through():
    inner = ThreadRecordingStore()
    store = ThreadConfinedStore(inner)
    try:
        with pytest.raises(MemoryUnavailable, match="HTTP 503"):
            store.ask("anything")
        assert store.bank_id == "dejafail-shopfront"
        assert store.repo == "shopfront"
        assert store.resets == 0
        store.reset()
        assert inner.resets == 1
        with pytest.raises(AttributeError):
            store.no_such_method
    finally:
        store.shutdown()


def test_thread_confined_store_works_as_triage_memory():
    inner = ThreadRecordingStore()
    inner.record_failure(RUN, SIG)
    store = ThreadConfinedStore(inner)
    try:
        llm = FakeLLM([{"kind": "flaky", "confidence": 0.9, "summary": "Seen before.", "evidence_ids": ["m1"]}])
        diagnosis = Triage(llm, store, "shopfront").diagnose(
            "FAILED tests/test_checkout.py::test_checkout_total - AssertionError: "
            "assert Decimal('0.00') == Decimal('54.97')\n"
        )
        assert diagnosis.verdict.used_memory is True
        assert inner.threads[-1].startswith("hindsight")
    finally:
        store.shutdown()
