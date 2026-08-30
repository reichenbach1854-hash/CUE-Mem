"""Runtime-only redaction helpers for provider error messages."""

from __future__ import annotations

import os
import re


_CREDENTIAL_ENV_NAMES = (
    "CUE_MEM_LLM_API_KEY",
    "CUE_MEM_LLM_BASE_URL",
    "CUE_MEM_AUDIO_API_KEY",
    "CUE_MEM_AUDIO_BASE_URL",
    "CUE_MEM_VLLM_API_KEY",
    "CUE_MEM_VLLM_BASE_URL",
    "AMEM_API_KEY",
    "AMEM_API_BASE",
    "MEMORYOS_API_KEY",
    "MEMORYOS_API_BASE",
    "CUE_MEM_IMAGE_AIFAST_API_KEY",
    "CUE_MEM_IMAGE_AIFAST_BASE_URL",
)


def redact_runtime_text(value: object) -> str:
    """Redact configured credentials and URL-shaped request metadata."""

    text = str(value)
    for name in _CREDENTIAL_ENV_NAMES:
        configured = os.environ.get(name, "").strip()
        if configured:
            text = text.replace(configured, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|authorization|token)(\s*[=:]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        text,
    )
    text = re.sub(r"https?://[^\s'\"\]\[)>,;]+", "[REDACTED_URL]", text)
    return text


def safe_runtime_error(prefix: str, error: BaseException) -> RuntimeError:
    """Build an exception whose message cannot disclose runtime credentials."""

    return RuntimeError(f"{prefix}: {redact_runtime_text(error)}")
