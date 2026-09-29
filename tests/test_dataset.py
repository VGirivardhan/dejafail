import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

from dejafail.models import VERDICT_KINDS
from dejafail.signature import extract_signature

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_dataset.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_is_deterministic_and_sized():
    mod = _load()
    runs_a, demos_a = mod.build()
    runs_b, demos_b = mod.build()
    assert runs_a == runs_b and demos_a == demos_b
    assert len(runs_a) == 48
    assert set(demos_a) == {"01_checkout_flaky_again.log", "02_pydantic_again.log", "03_new_redis_outage.log"}
    assert [r["started_at"] for r in runs_a] == sorted(r["started_at"] for r in runs_a)


def test_labels_are_valid_and_balanced():
    runs, _ = _load().build()
    labels = Counter(r["truth_label"] for r in runs)
    assert set(labels) <= set(VERDICT_KINDS) - {"unknown"}
    assert labels["flaky"] == 17
    assert labels["regression"] == 10


def test_each_recurring_pattern_shares_one_signature():
    runs, _ = _load().build()
    by_pattern = defaultdict(set)
    for r in runs:
        by_pattern[r["pattern"]].add(extract_signature(r["log"]).sig_hash)
    for pattern in ["flaky_checkout", "flaky_inventory", "pydantic", "disk", "stripe", "npm", "discount", "timeout"]:
        assert len(by_pattern[pattern]) == 1, pattern


def test_keyerror_regression_differs_from_flaky_checkout_but_same_test():
    runs, _ = _load().build()
    flaky = next(r for r in runs if r["pattern"] == "flaky_checkout")
    keyerr = next(r for r in runs if r["pattern"] == "keyerror")
    a, b = extract_signature(flaky["log"]), extract_signature(keyerr["log"])
    assert a.test_id == b.test_id == "tests/test_checkout.py::test_checkout_total"
    assert a.sig_hash != b.sig_hash


def test_demo_logs_match_history_or_are_new():
    runs, demos = _load().build()
    known = {extract_signature(r["log"]).sig_hash for r in runs}
    assert extract_signature(demos["01_checkout_flaky_again.log"]).sig_hash in known
    assert extract_signature(demos["02_pydantic_again.log"]).sig_hash in known
    assert extract_signature(demos["03_new_redis_outage.log"]).sig_hash not in known


def test_flaky_logs_do_not_leak_the_answer():
    runs, _ = _load().build()
    for r in runs:
        if r["truth_label"] == "flaky":
            text = r["log"].lower()
            assert "flaky" not in text and "rerun" not in text and "intermittent" not in text
