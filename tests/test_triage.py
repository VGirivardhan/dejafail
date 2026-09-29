import pytest

from dejafail.llm import LLMOutputError
from dejafail.memory import FakeStore, MemoryUnavailable
from dejafail.models import CIRun, Outcome
from dejafail.signature import extract_signature
from dejafail.triage import Triage, validate_verdict_json
from tests.fakes import FakeLLM

LOG = "FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert Decimal('0.00') == Decimal('54.97')\n"
GOOD = {"kind": "flaky", "confidence": 0.9, "summary": "Known flaky test.", "next_action": "Rerun.", "evidence_ids": ["m1", "m99"]}


def _seeded_store(times: int) -> FakeStore:
    store = FakeStore()
    sig = extract_signature(LOG)
    for i in range(times):
        run = CIRun(f"run-2{i:02d}", f"2026-09-1{i}T09:00:00+00:00", "main", "x64", "c" * 40, LOG, "flaky",
                    Outcome(True, "flaky", "Passed on rerun."))
        store.record_failure(run, sig)
        store.record_outcome(run, sig, run.outcome)
    return store


def test_stateless_prompt_has_no_memory_section():
    llm = FakeLLM([dict(GOOD, evidence_ids=[])])
    d = Triage(llm, _seeded_store(2)).diagnose(LOG, use_memory=False)
    assert "PAST MEMORIES" not in llm.prompts[0][1]
    assert d.verdict.used_memory is False
    assert d.verdict.evidence == []


def test_memory_prompt_lists_memories_and_maps_only_real_evidence():
    llm = FakeLLM([GOOD])
    d = Triage(llm, _seeded_store(2)).diagnose(LOG, use_memory=True)
    prompt = llm.prompts[0][1]
    assert "PAST MEMORIES" in prompt and "[m1]" in prompt and "Passed on rerun." in prompt
    assert d.verdict.kind == "flaky"
    assert len(d.verdict.evidence) == 1  # m99 was never recalled, so it is dropped
    assert d.verdict.seen_before_count == 2
    assert d.verdict.used_memory is True
    assert len(d.memories) == 4


def test_unseen_failure_says_so_in_prompt():
    llm = FakeLLM([dict(GOOD, evidence_ids=[])])
    d = Triage(llm, FakeStore()).diagnose(LOG, use_memory=True)
    assert "has not been seen before" in llm.prompts[0][1]
    assert d.verdict.seen_before_count == 0


def test_invalid_llm_output_becomes_unknown_verdict():
    llm = FakeLLM([LLMOutputError("no JSON object in model output")])
    d = Triage(llm, FakeStore()).diagnose(LOG)
    assert d.verdict.kind == "unknown"
    assert "no JSON object" in d.verdict.summary


def test_memory_outage_falls_back_to_stateless_with_error():
    class DownStore(FakeStore):
        def history(self, sig, limit=12):
            raise MemoryUnavailable("Hindsight Cloud is out of credits (402).")

    llm = FakeLLM([dict(GOOD, evidence_ids=[])])
    d = Triage(llm, DownStore()).diagnose(LOG, use_memory=True)
    assert "PAST MEMORIES" not in llm.prompts[0][1]
    assert d.verdict.used_memory is False
    assert "402" in d.verdict.memory_error


def test_confidence_is_clamped_and_kind_normalised():
    llm = FakeLLM([dict(GOOD, kind=" Flaky ", confidence=7, evidence_ids=[])])
    d = Triage(llm, FakeStore()).diagnose(LOG, use_memory=False)
    assert d.verdict.kind == "flaky"
    assert d.verdict.confidence == 1.0


@pytest.mark.parametrize("bad", [{"kind": "maybe", "summary": "x"}, {"kind": "flaky", "summary": ""},
                                 {"kind": "flaky", "summary": "x", "confidence": "high"}])
def test_validate_rejects_bad_objects(bad):
    with pytest.raises(ValueError):
        validate_verdict_json(bad)


def test_malformed_evidence_ids_int_returns_unknown():
    llm = FakeLLM([LLMOutputError("validation failed"), dict(GOOD, evidence_ids=[])])
    d = Triage(llm, FakeStore()).diagnose(LOG)
    assert d.verdict.kind == "unknown"


def test_evidence_ids_as_string_maps_to_one_item():
    llm = FakeLLM([dict(GOOD, evidence_ids="m1")])
    d = Triage(llm, _seeded_store(2)).diagnose(LOG, use_memory=True)
    assert len(d.verdict.evidence) == 1
    assert "Passed on rerun." in d.verdict.evidence[0].text


def test_validate_rejects_nan_confidence():
    bad = {"kind": "flaky", "summary": "x", "confidence": float("nan")}
    with pytest.raises(ValueError):
        validate_verdict_json(bad)


def test_next_action_null_yields_default_text():
    llm = FakeLLM([dict(GOOD, next_action=None, evidence_ids=[])])
    d = Triage(llm, FakeStore()).diagnose(LOG, use_memory=False)
    assert d.verdict.next_action == "Inspect the log excerpt."


def test_unavailable_llm_sets_unavailable_llm_error():
    from dejafail.llm import LLMUnavailable

    llm = FakeLLM([LLMUnavailable("Groq unreachable: timed out")])
    d = Triage(llm, FakeStore()).diagnose(LOG)
    assert d.verdict.kind == "unknown"
    assert d.verdict.llm_error == "unavailable: Groq unreachable: timed out"
    assert d.verdict.summary == "Could not get a verdict: Groq unreachable: timed out"


def test_invalid_output_sets_invalid_output_llm_error():
    llm = FakeLLM([LLMOutputError("no JSON object in model output")])
    d = Triage(llm, FakeStore()).diagnose(LOG)
    assert d.verdict.kind == "unknown"
    assert d.verdict.llm_error == "invalid output: no JSON object in model output"


def test_other_llm_errors_count_as_unavailable():
    from dejafail.llm import LLMError

    llm = FakeLLM([LLMError("something odd")])
    d = Triage(llm, FakeStore()).diagnose(LOG, use_memory=False)
    assert d.verdict.llm_error == "unavailable: something odd"


def test_llm_error_and_summary_text_are_truncated_to_300_chars():
    from dejafail.llm import LLMUnavailable

    llm = FakeLLM([LLMUnavailable("x" * 1000)])
    d = Triage(llm, FakeStore()).diagnose(LOG)
    assert d.verdict.llm_error.startswith("unavailable: x")
    assert len(d.verdict.llm_error) == len("unavailable: ") + 300
    assert d.verdict.summary.startswith("Could not get a verdict: x")
    assert len(d.verdict.summary) == len("Could not get a verdict: ") + 300


def test_successful_verdict_has_no_llm_error():
    llm = FakeLLM([dict(GOOD, evidence_ids=[])])
    d = Triage(llm, FakeStore()).diagnose(LOG, use_memory=False)
    assert d.verdict.llm_error is None
