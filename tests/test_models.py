from datetime import datetime, timezone

from dejafail.models import VERDICT_KINDS, CIRun, Outcome


def _run_dict():
    return {
        "run_id": "run-201",
        "started_at": "2026-09-07T09:14:00+00:00",
        "branch": "main",
        "runner": "gh-ubuntu-x64",
        "commit": "a" * 40,
        "log": "FAILED tests/x.py::t - AssertionError: boom",
        "truth_label": "flaky",
        "outcome": {"rerun_passed": True, "label": "flaky", "fix_note": "rerun passed"},
    }


def test_verdict_kinds_exact():
    assert VERDICT_KINDS == ("flaky", "regression", "dependency", "infra", "unknown")


def test_cirun_round_trip():
    run = CIRun.from_dict(_run_dict())
    assert run.outcome == Outcome(rerun_passed=True, label="flaky", fix_note="rerun passed")
    assert CIRun.from_dict(run.to_dict()) == run


def test_adhoc_run_id_is_stable_for_same_log():
    now = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)
    a = CIRun.adhoc("same log", now=now)
    b = CIRun.adhoc("same log", now=now)
    assert a.run_id == b.run_id
    assert a.run_id.startswith("manual-")
    assert a.started_at == "2026-09-29T10:00:00+00:00"
