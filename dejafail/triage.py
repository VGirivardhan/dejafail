"""The DejaFail agent: failure signature, then Hindsight recall, then an LLM verdict."""
from __future__ import annotations

import math
from typing import Any, Sequence

from .llm import LLM, LLMError, LLMOutputError, LLMUnavailable
from .memory import MemoryStore, MemoryUnavailable
from .models import VERDICT_KINDS, Diagnosis, Evidence, FailureSignature, Memory, Verdict
from .signature import extract_signature

ERROR_TEXT_LIMIT = 300  # characters of an LLM error kept in a verdict's summary and llm_error
UNAVAILABLE = "unavailable: "  # Verdict.llm_error prefix: Groq could not answer (network, rate limit, 4xx/5xx)
INVALID_OUTPUT = "invalid output: "  # Verdict.llm_error prefix: the model never produced a valid verdict
RECALL_LIMIT = 40  # memories recalled per failure; all of them feed the seen-before count
PROMPT_MEMORIES = 12  # of those, how many go into the prompt (Groq free tier: 8K tokens/minute)

SYSTEM_PROMPT = """You are DejaFail, a CI failure triage assistant for the repository "{repo}".
Classify the failing CI run into exactly one kind:
- flaky: the same test fails intermittently and passes on rerun without any code change
- regression: a code change broke behaviour; it needs a code fix
- dependency: a package, version or lockfile change broke the build
- infra: runner or environment problem (disk, network, DNS, OOM, timeouts, missing services)
- unknown: not enough information to decide
If PAST MEMORIES are provided, treat them as the history of this repository and weigh them heavily:
a failure that passed on rerun several times before is flaky; a failure that a code change fixed is not.
Reply with ONLY one JSON object, no prose:
{"kind": "flaky|regression|dependency|infra|unknown", "confidence": 0.0, "summary": "one sentence", "next_action": "one concrete step", "evidence_ids": ["m1"]}
Only cite ids that appear in PAST MEMORIES. If there are none, evidence_ids must be []."""


def build_user_prompt(sig: FailureSignature, aliases: dict[str, Memory] | None) -> str:
    lines = [
        "FAILURE SIGNATURE",
        f"test: {sig.test_id or '(none: job-level failure)'}",
        f"error: {sig.error_type}: {sig.message}",
        f"signature: {sig.sig_hash}",
        "",
        "LOG EXCERPT",
        sig.excerpt or "(empty log)",
    ]
    if aliases is not None:
        lines += ["", "PAST MEMORIES (earlier CI runs of this repository)"]
        if aliases:
            lines += [f"[{alias}] {m.date or 'undated'} | {m.text}" for alias, m in aliases.items()]
        else:
            lines.append("(none: this failure has not been seen before)")
    return "\n".join(lines)


def validate_verdict_json(obj: dict[str, Any]) -> None:
    kind = str(obj.get("kind", "")).strip().lower()
    if kind not in VERDICT_KINDS:
        raise ValueError(f"kind must be one of {', '.join(VERDICT_KINDS)}, got {kind!r}")
    try:
        conf = float(obj.get("confidence", 0))
        if not math.isfinite(conf):
            raise ValueError("confidence must be finite (not NaN or inf)")
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if not str(obj.get("summary", "")).strip():
        raise ValueError("summary must be a non-empty string")
    evidence_ids = obj.get("evidence_ids")
    if evidence_ids is not None and not isinstance(evidence_ids, (list, str)):
        raise ValueError("evidence_ids must be a list or string, or absent")


def verdict_from_json(obj: dict[str, Any], aliases: dict[str, Memory]) -> Verdict:
    evidence: list[Evidence] = []
    evidence_ids = obj.get("evidence_ids") or []
    if isinstance(evidence_ids, str):
        evidence_ids = [evidence_ids]
    for ref in evidence_ids:
        memory = aliases.get(str(ref).strip().strip("[]"))
        if memory is not None and all(e.memory_id != memory.id for e in evidence):
            evidence.append(Evidence(memory_id=memory.id, date=memory.date, text=memory.text))
    return Verdict(
        kind=str(obj["kind"]).strip().lower(),
        confidence=min(max(float(obj.get("confidence", 0)), 0.0), 1.0),
        summary=str(obj["summary"]).strip(),
        next_action=str(obj.get("next_action") or "").strip() or "Inspect the log excerpt.",
        evidence=evidence,
    )


def clip(text: str, limit: int = ERROR_TEXT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def failed_verdict(exc: Exception, prefix: str) -> Verdict:
    """An unknown verdict that says why the LLM gave none; prefix is "unavailable: " or "invalid output: "."""
    text = clip(str(exc))
    return Verdict(
        kind="unknown",
        confidence=0.0,
        summary=f"Could not get a verdict: {text}",
        next_action="Retry, or read the log excerpt manually.",
        llm_error=f"{prefix}{text}",
    )


def count_seen(memories: Sequence[Memory]) -> int:
    """Distinct earlier runs that share this failure's signature or test.

    A run's failure and its outcome ("<run_id>-outcome") count once; memories without a run id are skipped.
    """
    return len({m.run_id.removesuffix("-outcome") for m in memories if m.exact and m.run_id})


class Triage:
    def __init__(self, llm: LLM, memory: MemoryStore | None, repo: str = "shopfront"):
        self.llm = llm
        self.memory = memory
        self.system = SYSTEM_PROMPT.replace("{repo}", repo)

    def diagnose(self, log: str, use_memory: bool = True) -> Diagnosis:
        sig = extract_signature(log)
        memories: list[Memory] = []
        memory_error: str | None = None
        if use_memory:
            if self.memory is None:
                memory_error = "memory disabled"
            else:
                try:
                    memories = self.memory.history(sig, limit=RECALL_LIMIT)
                except MemoryUnavailable as exc:
                    memory_error = str(exc)
        aliases = {f"m{i}": m for i, m in enumerate(memories[:PROMPT_MEMORIES], start=1)}
        with_memory = use_memory and memory_error is None
        prompt = build_user_prompt(sig, aliases if with_memory else None)
        try:
            obj = self.llm.complete_json(self.system, prompt, validate=validate_verdict_json)
            verdict = verdict_from_json(obj, aliases)
        except LLMUnavailable as exc:
            verdict = failed_verdict(exc, UNAVAILABLE)
        except LLMOutputError as exc:
            verdict = failed_verdict(exc, INVALID_OUTPUT)
        except LLMError as exc:
            verdict = failed_verdict(exc, UNAVAILABLE)
        verdict.used_memory = with_memory
        verdict.memory_error = memory_error
        verdict.seen_before_count = count_seen(memories)
        return Diagnosis(sig=sig, verdict=verdict, memories=tuple(memories))
