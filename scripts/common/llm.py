"""Configuration helpers for OpenAI-compatible providers.

The repository deliberately contains no provider key or endpoint.  Callers
must provide credentials at runtime, normally with environment variables.
"""

from __future__ import annotations

import os
from typing import Any


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def required_env(name: str) -> str:
    value = env_value(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "CUE_MEM_LLM_API_KEY",
    base_url_env: str = "CUE_MEM_LLM_BASE_URL",
    timeout: float | None = None,
    default_headers: dict[str, str] | None = None,
) -> Any:
    """Build an OpenAI-compatible client using runtime-only configuration."""

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError("install the optional `openai` dependency to use an LLM script") from exc

    key = (api_key if api_key is not None else env_value(api_key_env)) or ""
    if not key:
        raise RuntimeError(f"missing API key; set {api_key_env} or pass a runtime value")

    endpoint = base_url if base_url is not None else env_value(base_url_env)
    kwargs: dict[str, Any] = {"api_key": key}
    if endpoint:
        kwargs["base_url"] = endpoint
    if timeout is not None:
        kwargs["timeout"] = timeout
    if default_headers:
        kwargs["default_headers"] = default_headers
    return OpenAI(**kwargs)


def usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", item.get("content", "")) or ""))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return "" if content is None else str(content)
