import io
import sys

import pytest

from dejafail import cli
from dejafail.config import ConfigError
from dejafail.models import Diagnosis, Evidence, FailureSignature, Verdict

SIG = FailureSignature("tests/t.py::t", "AssertionError", "boom", "boom", "abc123def456", "")


def test_format_diagnosis_with_memory_evidence():
    verdict = Verdict("flaky", 0.92, "Known flaky.", "Rerun.", [Evidence("f1", "2026-09-10", "passed on rerun")],
                      seen_before_count=4, used_memory=True)
    text = cli.format_diagnosis("DejaFail + memory", Diagnosis(SIG, verdict))
    assert "FLAKY" in text and "92%" in text and "seen before: 4" in text and "[2026-09-10] passed on rerun" in text


def test_format_diagnosis_shows_memory_error():
    verdict = Verdict("unknown", 0.0, "s", "a", memory_error="Hindsight Cloud is out of credits (402).")
    assert "402" in cli.format_diagnosis("x", Diagnosis(SIG, verdict))


def test_invalid_label_rejected():
    with pytest.raises(SystemExit):
        cli.main(["feedback", "x.log", "--label", "maybe"])


def test_config_error_returns_exit_code_2(monkeypatch, tmp_path, capsys):
    log = tmp_path / "a.log"
    log.write_text("FAILED t.py::t - AssertionError: x\n", encoding="utf-8")

    def boom():
        raise ConfigError("Missing GROQ_API_KEY in .env")

    monkeypatch.setattr(cli, "build_services", boom)
    assert cli.main(["diagnose", str(log)]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_non_ascii_output_does_not_crash_on_cp1252(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    cli.configure_output()
    print("● CartTotals ━━ done")
    stream.flush()
    assert raw.getvalue()


def test_build_services_passes_reasoning_effort_to_groq(monkeypatch):
    from dejafail import config, llm, memory

    cfg = config.load_config({"HINDSIGHT_API_KEY": "hsk_x", "GROQ_API_KEY": "gsk_y", "GROQ_REASONING_EFFORT": "medium"})
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(memory.HindsightStore, "from_config", classmethod(lambda cls, c: "store"))
    seen = {}

    class RecordingLLM:
        def __init__(self, *args, **kwargs):
            seen["args"], seen["kwargs"] = args, kwargs

    monkeypatch.setattr(llm, "GroqLLM", RecordingLLM)
    got_cfg, store, triage = cli.build_services()
    assert got_cfg is cfg and store == "store"
    assert seen["args"][:2] == ("gsk_y", "openai/gpt-oss-120b")
    assert seen["kwargs"]["reasoning_effort"] == "medium"


def test_replay_pause_defaults_to_twenty_seconds():
    args = cli._parser().parse_args(["replay"])
    assert args.pause == 20.0


def _replay_setup(monkeypatch, tmp_path, replies, env_extra=None):
    import json

    from dejafail import config
    from dejafail.memory import FakeStore
    from dejafail.models import CIRun, Outcome
    from dejafail.triage import Triage
    from tests.fakes import FakeLLM

    log = "FAILED tests/test_checkout.py::test_checkout_total - AssertionError: assert 0 == 1\n"
    runs = tmp_path / "runs.jsonl"
    rows = [CIRun(f"run-{i}", f"2026-09-0{i}T09:00:00+00:00", "main", "x64", "c" * 40, log, "flaky",
                  Outcome(True, "flaky", "rerun passed")).to_dict() for i in (1, 2)]
    runs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    env = {"HINDSIGHT_API_KEY": "hsk_x", "GROQ_API_KEY": "gsk_y", **(env_extra or {})}
    cfg = config.load_config(env)
    store = FakeStore()
    monkeypatch.setattr(cli, "build_services", lambda: (cfg, store, Triage(FakeLLM(replies), store)))
    return runs, tmp_path / "results.json"


def _verdict_json(kind):
    return {"kind": kind, "confidence": 0.8, "summary": "s", "next_action": "a", "evidence_ids": []}


def test_replay_success_writes_meta_and_reports_invalid_outputs(monkeypatch, tmp_path, capsys):
    from datetime import datetime

    from dejafail.llm import LLMOutputError
    from dejafail.replay import load_meta, load_steps

    replies = [LLMOutputError("no JSON object in model output"), _verdict_json("flaky"),
               _verdict_json("flaky"), _verdict_json("regression")]
    runs, out = _replay_setup(monkeypatch, tmp_path, replies)
    code = cli.main(["replay", "--runs", str(runs), "--out", str(out), "--pause", "0"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "accuracy: memory 50% vs stateless 50% over 2 runs" in printed
    assert "invalid-output verdicts: memory 1, stateless 0" in printed
    meta = load_meta(out)
    assert meta["model"] == "openai/gpt-oss-120b"
    assert meta["reasoning_effort"] == "low"
    assert meta["pause"] == 0.0
    assert meta["runs"] == 2
    assert meta["completed"] is True
    assert meta["aborted_reason"] is None
    assert datetime.fromisoformat(meta["generated_at"]).utcoffset().total_seconds() == 0
    steps = load_steps(out)
    assert len(steps) == 2 and steps[0].memory_llm_error.startswith("invalid output: ")


def test_replay_abort_keeps_partial_results_and_exits_3(monkeypatch, tmp_path, capsys):
    import json

    from dejafail.llm import LLMUnavailable
    from dejafail.replay import load_meta, load_steps

    replies = [_verdict_json("flaky"), _verdict_json("flaky"),
               LLMUnavailable("Groq request failed: 429 rate limit"), _verdict_json("flaky")]
    runs, out = _replay_setup(monkeypatch, tmp_path, replies)
    code = cli.main(["replay", "--runs", str(runs), "--out", str(out), "--pause", "0"])
    captured = capsys.readouterr()
    assert code == 3
    assert "replay stopped at run 2: " in captured.err
    assert "429 rate limit" in captured.err
    assert f"Partial results (1 runs) saved to {out}." in captured.err
    assert "accuracy:" not in captured.out
    assert [s.run_id for s in load_steps(out)] == ["run-1"]
    meta = load_meta(out)
    assert meta["completed"] is False
    assert "429 rate limit" in meta["aborted_reason"]
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["runs"] == 2


def test_replay_meta_reasoning_effort_is_null_for_models_that_do_not_take_it(monkeypatch, tmp_path, capsys):
    from dejafail.replay import load_meta

    replies = [_verdict_json("flaky")] * 4
    runs, out = _replay_setup(monkeypatch, tmp_path, replies, {"GROQ_MODEL": "qwen/qwen3-32b"})
    assert cli.main(["replay", "--runs", str(runs), "--out", str(out), "--pause", "0"]) == 0
    capsys.readouterr()
    meta = load_meta(out)
    assert meta["model"] == "qwen/qwen3-32b"
    assert meta["reasoning_effort"] is None
