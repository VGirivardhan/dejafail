"""Command line entry point: python -m dejafail <command>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .models import VERDICT_KINDS, CIRun, Diagnosis, Outcome

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "shopfront"
DEFAULT_RUNS = DATA_DIR / "runs.jsonl"
DEFAULT_RESULTS = DATA_DIR / "replay_results.json"


def configure_output() -> None:
    """Never crash on a legacy Windows console when logs contain characters like ● or ━."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def format_diagnosis(title: str, diagnosis: Diagnosis) -> str:
    v = diagnosis.verdict
    lines = [
        f"== {title} ==",
        f"verdict:     {v.kind.upper()}  (confidence {v.confidence:.0%})",
        f"summary:     {v.summary}",
        f"next action: {v.next_action}",
    ]
    if v.used_memory:
        lines.append(f"seen before: {v.seen_before_count} earlier run(s)")
        lines += [f"  - [{e.date or 'undated'}] {e.text}" for e in v.evidence]
    if v.memory_error:
        lines.append(f"memory:      unavailable ({v.memory_error})")
    return "\n".join(lines)


def build_services() -> tuple[Any, Any, Any]:
    from .config import load_config
    from .llm import GroqLLM
    from .memory import HindsightStore
    from .triage import Triage

    cfg = load_config()
    store = HindsightStore.from_config(cfg)
    return cfg, store, Triage(GroqLLM(cfg.groq_api_key, cfg.groq_model), store, cfg.repo)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dejafail", description="CI triage agent that remembers every red build.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("diagnose", help="Diagnose a failing CI log")
    p.add_argument("log", type=Path)
    p.add_argument("--no-memory", action="store_true", help="stateless baseline only")
    p.add_argument("--compare", action="store_true", help="show stateless and memory verdicts")

    p = sub.add_parser("feedback", help="Teach DejaFail the real outcome of a failure")
    p.add_argument("log", type=Path)
    p.add_argument("--label", required=True, choices=VERDICT_KINDS)
    p.add_argument("--note", default="")
    p.add_argument("--rerun-passed", action="store_true")

    p = sub.add_parser("seed", help="Load CI history into memory without calling the LLM")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--until", help="only runs on or before this date (YYYY-MM-DD)")

    p = sub.add_parser("replay", help="Replay CI history and measure accuracy with and without memory")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--limit", type=int)
    p.add_argument("--pause", type=float, default=0.0, help="seconds between runs (Groq free tier)")

    p = sub.add_parser("ask", help="Ask the memory bank a question (Hindsight reflect)")
    p.add_argument("question")
    return parser


def _read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _run(args: argparse.Namespace) -> int:
    from .memory import MemoryUnavailable
    from .replay import load_runs, replay, save_steps, seed
    from .signature import extract_signature

    cfg, store, triage = build_services()
    if args.command == "diagnose":
        log = _read_log(args.log)
        if args.compare or args.no_memory:
            print(format_diagnosis("Stateless LLM", triage.diagnose(log, use_memory=False)))
        if not args.no_memory:
            if args.compare:
                print()
            print(format_diagnosis("DejaFail + Hindsight memory", triage.diagnose(log, use_memory=True)))
        return 0
    if args.command == "feedback":
        log = _read_log(args.log)
        run, sig = CIRun.adhoc(log), extract_signature(log)
        store.ensure_bank()
        store.record_failure(run, sig)
        store.record_outcome(run, sig, Outcome(args.rerun_passed, args.label, args.note))
        print(f"Learned: {sig.test_id or 'job failure'} [{sig.sig_hash}] -> {args.label}")
        return 0
    if args.command == "seed":
        store.reset()
        count = seed(load_runs(args.runs), store, until=args.until)
        print(f"Seeded {count} runs into {store.bank_id}")
        _refresh(store, MemoryUnavailable)
        return 0
    if args.command == "replay":
        runs = load_runs(args.runs)[: args.limit] if args.limit else load_runs(args.runs)
        done = []

        def on_step(step):
            done.append(step)
            save_steps(done, args.out)
            print(f"[{step.index:02d}/{len(runs)}] {step.run_id} truth={step.truth:<10} "
                  f"memory={step.memory_kind:<10} stateless={step.stateless_kind:<10} seen={step.seen_before}")

        steps = replay(runs, triage, store, on_step=on_step, pause=args.pause)
        mem = sum(s.memory_correct for s in steps) / max(len(steps), 1)
        base = sum(s.stateless_correct for s in steps) / max(len(steps), 1)
        print(f"accuracy: memory {mem:.0%} vs stateless {base:.0%} over {len(steps)} runs -> {args.out}")
        _refresh(store, MemoryUnavailable)
        return 0
    if args.command == "ask":
        print(store.ask(args.question))
        return 0
    return 1


def _refresh(store: Any, unavailable: type[Exception]) -> None:
    try:
        store.refresh_summary()
        print("Requested refresh of the 'Flaky ledger' mental model.")
    except unavailable as exc:
        print(f"warning: could not refresh mental model: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    configure_output()
    args = _parser().parse_args(argv)
    from .config import ConfigError
    from .llm import LLMError
    from .memory import MemoryUnavailable

    try:
        return _run(args)
    except (ConfigError, MemoryUnavailable, LLMError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
