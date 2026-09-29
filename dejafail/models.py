"""Plain data types shared by every DejaFail module."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

VERDICT_KINDS: tuple[str, ...] = ("flaky", "regression", "dependency", "infra", "unknown")


@dataclass(frozen=True)
class FailureSignature:
    test_id: str | None
    error_type: str
    message: str
    normalized: str
    sig_hash: str
    excerpt: str


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    date: str | None = None
    tags: tuple[str, ...] = ()
    run_id: str | None = None
    exact: bool = False  # True when it came from the signature/test tag-filtered recall


@dataclass(frozen=True)
class Evidence:
    memory_id: str
    date: str | None
    text: str


@dataclass
class Verdict:
    kind: str
    confidence: float
    summary: str
    next_action: str
    evidence: list[Evidence] = field(default_factory=list)
    seen_before_count: int = 0
    used_memory: bool = False
    memory_error: str | None = None


@dataclass(frozen=True)
class Diagnosis:
    sig: FailureSignature
    verdict: Verdict
    memories: tuple[Memory, ...] = ()


@dataclass(frozen=True)
class Outcome:
    rerun_passed: bool
    label: str
    fix_note: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Outcome":
        return cls(
            rerun_passed=bool(data["rerun_passed"]),
            label=str(data["label"]),
            fix_note=str(data.get("fix_note", "")),
        )


@dataclass(frozen=True)
class CIRun:
    run_id: str
    started_at: str
    branch: str
    runner: str
    commit: str
    log: str
    truth_label: str
    outcome: Outcome

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CIRun":
        return cls(
            run_id=str(data["run_id"]),
            started_at=str(data["started_at"]),
            branch=str(data["branch"]),
            runner=str(data["runner"]),
            commit=str(data["commit"]),
            log=str(data["log"]),
            truth_label=str(data["truth_label"]),
            outcome=Outcome.from_dict(data["outcome"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def adhoc(cls, log: str, now: datetime | None = None) -> "CIRun":
        """A run for a pasted log that did not come from the dataset."""
        now = now or datetime.now(timezone.utc)
        digest = hashlib.sha1(log.encode("utf-8", "replace")).hexdigest()[:8]
        return cls(
            run_id=f"manual-{digest}",
            started_at=now.isoformat(timespec="seconds"),
            branch="unknown",
            runner="unknown",
            commit="unknown",
            log=log,
            truth_label="unknown",
            outcome=Outcome(rerun_passed=False, label="unknown", fix_note=""),
        )
