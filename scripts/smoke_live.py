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

ROOT = Path(__file__).resolve().parent.parent
LOG = "FAILED tests/test_smoke.py::test_ping - AssertionError: assert 'pong' == 'ping'\n"
DEMO_LOG = ROOT / "data" / "shopfront" / "demo_logs" / "01_checkout_flaky_again.log"
SMOKE_BANK = "dejafail-smoke"


def print_unknown(label: str, v) -> None:
    """Show why a verdict came back unknown before the run is declared failed."""
    print(f"{label} summary: {v.summary}")
    if v.llm_error:
        print(f"{label} llm_error: {v.llm_error}")


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
            llm = GroqLLM(cfg.groq_api_key, cfg.groq_model, reasoning_effort=cfg.groq_reasoning_effort)
            triage = Triage(llm, store, cfg.repo)
            d = triage.diagnose(LOG)
            v = d.verdict
            print(f"verdict={v.kind} confidence={v.confidence:.2f} seen_before={v.seen_before_count} evidence={len(v.evidence)}")
            print(f"reflect: {store.ask('Is test_ping flaky?')[:200]!r}")
            # A realistic-size prompt must fit the model's limits too (stateless: no memory needed).
            demo = triage.diagnose(DEMO_LOG.read_text(encoding="utf-8"), use_memory=False).verdict
            print(
                f"demo log {DEMO_LOG.name} (stateless): verdict={demo.kind} "
                f"confidence={demo.confidence:.2f} summary={demo.summary[:160]!r}"
            )

            problems = []
            if len(memories) == 0:
                problems.append("no memories recalled")
            if not any(m.exact for m in memories):
                problems.append("no exact (tag-matched) memory recalled")
            if not v.used_memory:
                problems.append("v.used_memory is False")
            if v.memory_error:
                problems.append("v.memory_error is set")
            if v.seen_before_count < 1:
                problems.append("v.seen_before_count is 0")
            if v.kind == "unknown":
                print_unknown("smoke verdict", v)
                problems.append("v.kind is unknown")
            if demo.kind == "unknown":
                print_unknown("demo log verdict", demo)
                problems.append(f"demo log {DEMO_LOG.name} verdict is unknown")

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
