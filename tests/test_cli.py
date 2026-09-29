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
