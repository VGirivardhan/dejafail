from datetime import datetime

import pytest

from dejafail.memory import (
    MENTAL_MODEL_ID,
    FakeStore,
    HindsightStore,
    MemoryUnavailable,
    failure_text,
    outcome_text,
)
from dejafail.models import CIRun, FailureSignature, Outcome
from tests.fakes import FakeHindsightClient, recall_result

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
