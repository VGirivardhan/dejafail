"""One live round trip against Hindsight Cloud and Groq. Run after filling in .env.

Uses a throwaway bank (dejafail-smoke) and deletes it at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dejafail.cli import configure_output  # noqa: E402
from dejafail.config import ConfigError, load_config  # noqa: E402
from dejafail.llm import GroqLLM, LLMError  # noqa: E402
from dejafail.memory import HindsightStore, MemoryUnavailable  # noqa: E402
from dejafail.models import CIRun, Outcome  # noqa: E402
from dejafail.signature import extract_signature  # noqa: E402
from dejafail.triage import Triage  # noqa: E402

LOG = "FAILED tests/test_smoke.py::test_ping - AssertionError: assert 'pong' == 'ping'\n"
SMOKE_BANK = "dejafail-smoke"


def main() -> int:
    configure_output()
    try:
        cfg = load_config()
        store = HindsightStore.from_config(cfg, bank_id=SMOKE_BANK)
        try:
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

            problems = []
            if len(memories) == 0:
                problems.append("no memories recalled")
            if not v.used_memory:
                problems.append("v.used_memory is False")
            if v.memory_error:
                problems.append("v.memory_error is set")
            if v.kind == "unknown":
                problems.append("v.kind is unknown")

            if problems:
                print(f"SMOKE FAILED: {'; '.join(problems)}")
                return 1
            else:
                print("SMOKE OK")
                return 0
        finally:
            try:
                store._client.delete_bank(SMOKE_BANK)
            except Exception as exc:
                print(f"warning: could not delete bank {SMOKE_BANK}: {type(exc).__name__}", file=sys.stderr)
    except ConfigError as e:
        print(f"SMOKE FAILED: {e}")
        return 2
    except MemoryUnavailable as e:
        print(f"SMOKE FAILED: {e}")
        return 2
    except LLMError as e:
        print(f"SMOKE FAILED: {e}")
        return 2
    except OSError as e:
        print(f"SMOKE FAILED: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
