import json

from dejafail.memory import FakeStore
from dejafail.models import CIRun, Outcome
from dejafail.replay import ReplayStep, load_runs, load_steps, replay, rolling_accuracy, save_steps, seed
from dejafail.triage import Triage
from tests.fakes import FakeLLM

LOG = "FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert 0 == 1\n"


def _run(i: int, day: int) -> CIRun:
    return CIRun(f"run-{i}", f"2026-09-{day:02d}T09:00:00+00:00", "main", "x64", "c" * 40, LOG, "flaky",
                 Outcome(True, "flaky", "rerun passed"))


def _verdict(kind):
    return {"kind": kind, "confidence": 0.8, "summary": "s", "next_action": "a", "evidence_ids": []}


def test_replay_resets_bank_scores_and_records_every_run():
    store = FakeStore()
    store.records.append(object())  # leftover state must be cleared
    # per run: memory diagnosis first, then stateless diagnosis
    llm = FakeLLM([_verdict("regression"), _verdict("regression"), _verdict("flaky"), _verdict("regression")])
    steps = replay([_run(1, 7), _run(2, 8)], Triage(llm, store), store)
    assert store.resets == 1
    assert len(store.records) == 4
    assert [s.memory_correct for s in steps] == [False, True]
    assert [s.stateless_correct for s in steps] == [False, False]
    assert steps[1].seen_before == 1


def test_rolling_accuracy_window():
    steps = [ReplayStep(i, f"r{i}", "2026-09-07", None, "flaky", "flaky" if i > 2 else "infra", "infra", 0)
             for i in range(1, 5)]
    rows = rolling_accuracy(steps, window=2)
    assert [r["memory"] for r in rows] == [0.0, 0.0, 0.5, 1.0]
    assert [r["stateless"] for r in rows] == [0.0, 0.0, 0.0, 0.0]
    assert rows[-1]["run"] == 4


def test_load_runs_sorts_and_seed_until(tmp_path):
    path = tmp_path / "runs.jsonl"
    lines = [json.dumps(dict(_run(2, 9).to_dict(), pattern="x")), json.dumps(_run(1, 7).to_dict())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs = load_runs(path)
    assert [r.run_id for r in runs] == ["run-1", "run-2"]
    store = FakeStore()
    assert seed(runs, store, until="2026-09-08") == 1
    assert len(store.records) == 2


def test_steps_round_trip(tmp_path):
    steps = [ReplayStep(1, "r1", "2026-09-07T09:00:00+00:00", "t", "flaky", "flaky", "regression", 0)]
    save_steps(steps, tmp_path / "out.json")
    assert load_steps(tmp_path / "out.json") == steps


def test_eight_positional_fields_still_build_a_step_with_no_errors():
    step = ReplayStep(1, "r1", "2026-09-07", None, "flaky", "flaky", "infra", 0)
    assert (step.memory_error, step.memory_llm_error, step.stateless_llm_error) == (None, None, None)


def test_steps_round_trip_with_meta(tmp_path):
    from dejafail.replay import load_meta

    path = tmp_path / "out.json"
    steps = [ReplayStep(1, "r1", "2026-09-07T09:00:00+00:00", "t", "flaky", "flaky", "unknown", 0,
                        stateless_llm_error="invalid output: no JSON object in model output")]
    meta = {"model": "openai/gpt-oss-120b", "reasoning_effort": "low", "completed": True}
    save_steps(steps, path, meta)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["meta"] == meta
    assert len(raw["steps"]) == 1
    assert load_steps(path) == steps
    assert load_meta(path) == meta


def test_save_without_meta_writes_an_empty_meta_object(tmp_path):
    from dejafail.replay import load_meta

    path = tmp_path / "out.json"
    save_steps([], path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"meta": {}, "steps": []}
    assert load_meta(path) == {}
    assert load_steps(path) == []


def test_legacy_bare_list_results_file_still_loads(tmp_path):
    from dejafail.replay import load_meta

    path = tmp_path / "legacy.json"
    legacy = [{"index": 1, "run_id": "r1", "started_at": "2026-09-07T09:00:00+00:00", "test_id": "t",
               "truth": "flaky", "memory_kind": "flaky", "stateless_kind": "regression", "seen_before": 0}]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_steps(path) == [ReplayStep(1, "r1", "2026-09-07T09:00:00+00:00", "t", "flaky", "flaky", "regression", 0)]
    assert load_meta(path) == {}


def test_replay_aborted_carries_step_index_and_reason():
    from dejafail.replay import ReplayAborted

    exc = ReplayAborted(2, "memory arm: unavailable: Groq unreachable")
    assert isinstance(exc, RuntimeError)
    assert exc.step_index == 2
    assert exc.reason == "memory arm: unavailable: Groq unreachable"
    assert "Groq unreachable" in str(exc)
