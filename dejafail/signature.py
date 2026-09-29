"""Turn a raw CI log into a stable FailureSignature. Pure functions, no I/O."""
from __future__ import annotations

import hashlib
import re

from .models import FailureSignature

MAX_EXCERPT_CHARS = 2500
MAX_MESSAGE_CHARS = 500

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_GHA_TS_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?", re.M)

_PYTEST_SUMMARY = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?: - (.*))?$", re.M)
_JUNIT_FAIL = re.compile(r"^\[ERROR\]\s+(\w+)\(([\w.$]+)\).*<<< (?:FAILURE|ERROR)!", re.M)
_JEST_FAIL = re.compile(r"^\s*●\s+(.+?)\s*$", re.M)
_ERROR_LINE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception))\b:\s*(.*)")
_NPM_ERR = re.compile(r"npm ERR! code (\w+)")
_GHA_ERROR = re.compile(r"##\[error\](.+)")
_FAILURE_MARKER = re.compile(
    r"(^FAILED |^ERROR |Traceback|Error:|Exception:|●|<<< FAILURE|npm ERR!|##\[error\])"
)

_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<ts>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-]+)+[\\/]"), "<path>/"),
    (re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b"), "<hash>"),
    (re.compile(r"\bline \d+"), "line <n>"),
    (re.compile(r":\d+\b"), ":<n>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b"), "<dur>"),
    (re.compile(r"\s+"), " "),
]


def clean_log(log: str) -> str:
    """Normalise line endings, remove ANSI colour codes and GitHub Actions timestamp prefixes."""
    text = (log or "").replace("\r\n", "\n").replace("\r", "\n")
    return _GHA_TS_PREFIX.sub("", _ANSI.sub("", text))


def normalize(text: str) -> str:
    """Replace run-specific noise (addresses, paths, hashes, durations) with placeholders."""
    out = text
    for pattern, replacement in _NORMALIZERS:
        out = pattern.sub(replacement, out)
    return out.strip()


def _split_error(text: str) -> tuple[str, str]:
    match = _ERROR_LINE.search(text)
    if match:
        return match.group(1), match.group(2).strip()
    return "Failure", text.strip()


def _first_error(log: str) -> tuple[str, str]:
    match = _ERROR_LINE.search(log)
    if match:
        return match.group(1), match.group(2).strip()
    match = _NPM_ERR.search(log)
    if match:
        return "npm ERR", f"code {match.group(1)}"
    match = _GHA_ERROR.search(log)
    if match:
        return "CIError", match.group(1).strip()
    return "Unknown", ""


def _find_failure(log: str) -> tuple[str | None, str, str]:
    match = _PYTEST_SUMMARY.search(log)
    if match:
        rest = (match.group(2) or "").strip()
        error_type, message = _split_error(rest) if rest else _first_error(log)
        return match.group(1), error_type, message
    match = _JUNIT_FAIL.search(log)
    if match:
        return f"{match.group(2)}.{match.group(1)}", *_first_error(log)
    match = _JEST_FAIL.search(log)
    if match:
        return match.group(1), *_first_error(log)
    return None, *_first_error(log)


def _excerpt(log: str) -> str:
    lines = log.splitlines()
    first = next(
        (i for i, line in enumerate(lines) if _FAILURE_MARKER.search(line)),
        max(len(lines) - 60, 0),
    )
    chunk = "\n".join(lines[max(first - 15, 0) : first + 45])
    return chunk[:MAX_EXCERPT_CHARS]


def extract_signature(log: str) -> FailureSignature:
    text = clean_log(log)
    test_id, error_type, message = _find_failure(text)
    message = message[:MAX_MESSAGE_CHARS]
    normalized = normalize(message)
    sig_hash = hashlib.sha1(f"{error_type}|{normalized}".encode("utf-8")).hexdigest()[:12]
    return FailureSignature(
        test_id=test_id,
        error_type=error_type,
        message=message,
        normalized=normalized,
        sig_hash=sig_hash,
        excerpt=_excerpt(text),
    )
