"""Replay CI history in order to measure how memory changes triage accuracy."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .memory import MemoryStore, MemoryUnavailable
from .models import CIRun, Verdict
from .signature import extract_signature
from .triage import UNAVAILABLE, Triage


@dataclass(frozen=True)
class ReplayStep:
    index: int
    run_id: str
    started_at: str
    test_id: str | None
    truth: str
    memory_kind: str
    stateless_kind: str
    seen_before: int
    memory_error: str | None = None  # recall failed for the memory arm
    memory_llm_error: str | None = None  # "invalid output: ..." when the memory arm is scored as unknown
    stateless_llm_error: str | None = None  # same for the stateless arm

    @property
    def memory_correct(self) -> bool:
        return self.memory_kind == self.truth

    @property
    def stateless_correct(self) -> bool:
        return self.stateless_kind == self.truth


class ReplayAborted(RuntimeError):
    """The replay stopped because a run could not be scored honestly (recall or LLM unavailable)."""

    def __init__(self, step_index: int, reason: str):
        super().__init__(f"replay stopped at run {step_index}: {reason}")
        self.step_index = step_index
        self.reason = reason


def load_runs(path: str | Path) -> list[CIRun]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    runs = [CIRun.from_dict(json.loads(line)) for line in lines if line.strip()]
    return sorted(runs, key=lambda run: run.started_at)


def seed(runs: Sequence[CIRun], memory: MemoryStore, until: str | None = None) -> int:
    """Store failures and outcomes without calling the LLM. Returns how many runs were stored."""
    stored = 0
    for run in runs:
        if until and run.started_at[:10] > until:
            break
        sig = extract_signature(run.log)
        memory.record_failure(run, sig)
        memory.record_outcome(run, sig, run.outcome)
        stored += 1
    return stored


def _abort_reason(remembered: Verdict, stateless: Verdict) -> str | None:
    if remembered.memory_error:
        return f"memory arm: Hindsight recall failed: {remembered.memory_error}"
    for arm, verdict in (("memory arm", remembered), ("stateless arm", stateless)):
        if verdict.llm_error and verdict.llm_error.startswith(UNAVAILABLE):
            return f"{arm}: LLM {verdict.llm_error}"
    return None


def _scored_kind(verdict: Verdict) -> str:
    # Only "invalid output" LLM errors reach scoring (unavailable ones abort): the model failed, so unknown.
    return "unknown" if verdict.llm_error else verdict.kind


def replay(
    runs: Sequence[CIRun],
    triage: Triage,
    memory: MemoryStore,
    on_step: Callable[[ReplayStep], None] | None = None,
    pause: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ReplayStep]:
    """For each run: diagnose with and without memory, score both, then teach memory the outcome.

    Raises ReplayAborted when a run cannot be scored honestly: recall failed for the memory arm,
    either arm's LLM call was unavailable, or the run could not be taught to memory. An arm whose
    model output stayed invalid is scored as unknown and flagged in the step.
    """
    memory.reset()
    steps: list[ReplayStep] = []
    for index, run in enumerate(runs, start=1):
        remembered = triage.diagnose(run.log, use_memory=True)
        stateless = triage.diagnose(run.log, use_memory=False)
        reason = _abort_reason(remembered.verdict, stateless.verdict)
        if reason is not None:
            raise ReplayAborted(index, reason)
        try:
            memory.record_failure(run, remembered.sig)
            memory.record_outcome(run, remembered.sig, run.outcome)
        except MemoryUnavailable as exc:
            raise ReplayAborted(index, f"could not record run {run.run_id} in memory: {exc}") from exc
        step = ReplayStep(
            index=index,
            run_id=run.run_id,
            started_at=run.started_at,
            test_id=remembered.sig.test_id,
            truth=run.truth_label,
            memory_kind=_scored_kind(remembered.verdict),
            stateless_kind=_scored_kind(stateless.verdict),
            seen_before=remembered.verdict.seen_before_count,
            memory_error=remembered.verdict.memory_error,
            memory_llm_error=remembered.verdict.llm_error,
            stateless_llm_error=stateless.verdict.llm_error,
        )
        steps.append(step)
        if on_step is not None:
            on_step(step)
        if pause:
            sleep(pause)
    return steps


def rolling_accuracy(steps: Sequence[ReplayStep], window: int = 8) -> list[dict[str, float]]:
    rows = []
    for i, step in enumerate(steps):
        chunk = steps[max(0, i - window + 1) : i + 1]
        rows.append({
            "run": step.index,
            "memory": sum(s.memory_correct for s in chunk) / len(chunk),
            "stateless": sum(s.stateless_correct for s in chunk) / len(chunk),
        })
    return rows


def save_steps(steps: Sequence[ReplayStep], path: str | Path, meta: dict[str, Any] | None = None) -> None:
    payload = {"meta": meta or {}, "steps": [asdict(s) for s in steps]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_results(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # legacy results file: a bare list of steps
        return {"meta": {}, "steps": data}
    return data


def load_steps(path: str | Path) -> list[ReplayStep]:
    return [ReplayStep(**item) for item in _load_results(path).get("steps", [])]


def load_meta(path: str | Path) -> dict[str, Any]:
    return dict(_load_results(path).get("meta") or {})
