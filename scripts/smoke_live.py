"""One live round trip against Hindsight Cloud and Groq. Run after filling in .env.

Uses a throwaway bank (dejafail-smoke) and deletes it at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dejafail.config import load_config  # noqa: E402
from dejafail.llm import GroqLLM  # noqa: E402
from dejafail.memory import HindsightStore  # noqa: E402
from dejafail.models import CIRun, Outcome  # noqa: E402
from dejafail.signature import extract_signature  # noqa: E402
from dejafail.triage import Triage  # noqa: E402

LOG = "FAILED tests/test_smoke.py::test_ping - AssertionError: assert 'pong' == 'ping'\n"


def main() -> int:
    cfg = load_config()
    store = HindsightStore.from_config(cfg, bank_id="dejafail-smoke")
    store.reset()
    run, sig = CIRun.adhoc(LOG), extract_signature(LOG)
    store.record_failure(run, sig)
    store.record_outcome(run, sig, Outcome(rerun_passed=True, label="flaky", fix_note="smoke test"))
    memories = store.history(sig)
    print(f"recall returned {len(memories)} memories")
    for m in memories:
        print(f"  exact={m.exact} date={m.date} run_id={m.run_id} tags={list(m.tags)} text={m.text[:90]!r}")
    triage = Triage(GroqLLM(cfg.groq_api_key, cfg.groq_model), store, cfg.repo)
    d = triage.diagnose(LOG)
    v = d.verdict
    print(f"verdict={v.kind} confidence={v.confidence:.2f} seen_before={v.seen_before_count} evidence={len(v.evidence)}")
    print(f"reflect: {store.ask('Is test_ping flaky?')[:200]!r}")
    store._client.delete_bank("dejafail-smoke")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
