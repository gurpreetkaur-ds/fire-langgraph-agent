"""Redaction of secrets and sensitive personal data before anything is persisted or served."""
from __future__ import annotations

import os
import re
from typing import Any, List

REDACTED = "[REDACTED]"

# (label, pattern). Order matters: specific token formats before generic ones.
_SECRET_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("BEARER_TOKEN", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    ("FIRE_KEY", re.compile(r"\bfire_sk_[A-Za-z0-9_-]{6,}")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("DB_URL_CREDENTIALS", re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^@\s]+)@")),
    (
        "KEY_VALUE_SECRET",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|authorization)\b(\s*[:=]\s*)(['\"]?)([^\s'\",;]{4,})\3"
        ),
    ),
]

_PII_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)")),
]

_SENSITIVE_KEYS = re.compile(r"(?i)(api[_-]?key|secret|(?<![a-z])token(?!s)|password|passwd|authorization|credential|cookie|private[_-]?key)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _env_secret_values() -> List[str]:
    """Literal values of secret-looking environment variables, so exact leaks are always caught."""
    values = []
    for name, value in os.environ.items():
        if value and len(value) >= 8 and _SENSITIVE_KEYS.search(name):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


# Formats precise enough to treat as a leak. Generic `password=...` pairs are redacted for storage but are NOT
# evidence of a leak (educational answers and code samples legitimately contain them).
_HIGH_CONFIDENCE = {"PRIVATE_KEY", "BEARER_TOKEN", "FIRE_KEY", "API_KEY", "GITHUB_TOKEN", "AWS_ACCESS_KEY", "JWT", "DB_URL_CREDENTIALS"}
# Stored text is already redacted, so a leak shows up as one of our own markers rather than the raw secret.
_MARKER = re.compile(r"\[REDACTED_(PRIVATE_KEY|BEARER_TOKEN|FIRE_KEY|API_KEY|GITHUB_TOKEN|AWS_ACCESS_KEY|JWT|ENV_SECRET)\]|\[REDACTED\]:\[REDACTED\]@")


def find_secrets(text: str) -> List[str]:
    """Labels of high-confidence secret types in `text`, whether raw or already replaced by a redaction marker
    (used by the safety evaluator)."""
    if not text:
        return []
    found = [label for label, pat in _SECRET_PATTERNS if label in _HIGH_CONFIDENCE and pat.search(text)]
    found += [m.group(1) or "DB_URL_CREDENTIALS" for m in _MARKER.finditer(text)]
    if any(v in text for v in _env_secret_values()):
        found.append("ENVIRONMENT_SECRET")
    return sorted(set(found))


def redact_text(text: str, *, pii: bool = True) -> str:
    if not text:
        return text
    for value in _env_secret_values():
        text = text.replace(value, "[REDACTED_ENV_SECRET]")
    for label, pat in _SECRET_PATTERNS:
        if label == "DB_URL_CREDENTIALS":
            text = pat.sub(lambda m: f"{m.group(1)}[REDACTED]:[REDACTED]@", text)
        elif label == "KEY_VALUE_SECRET":
            text = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}", text)
        else:
            text = pat.sub(f"[REDACTED_{label}]", text)
    if pii:
        for label, pat in _PII_PATTERNS:
            if label == "CREDIT_CARD":
                text = pat.sub(
                    lambda m: "[REDACTED_CREDIT_CARD]" if _luhn_ok(re.sub(r"\D", "", m.group(0))) else m.group(0), text
                )
            else:
                text = pat.sub(f"[REDACTED_{label}]", text)
    return text


def redact_obj(obj: Any, *, pii: bool = True) -> Any:
    """Recursively redact strings; values under secret-looking keys are replaced wholesale."""
    if isinstance(obj, str):
        return redact_text(obj, pii=pii)
    if isinstance(obj, dict):
        return {
            k: (REDACTED if isinstance(k, str) and _SENSITIVE_KEYS.search(k) else redact_obj(v, pii=pii))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, pii=pii) for v in obj]
    return obj


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"
