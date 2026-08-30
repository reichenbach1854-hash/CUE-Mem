"""Runtime configuration for the RQ3 scripts.

The source experiment was run from a private checkout with data, caches and
service credentials next to the code.  The public copy keeps those concerns
outside the repository: paths are project-relative by default and all
service credentials/endpoints must be supplied at runtime.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def env_value(name: str, default: str | None = None) -> str | None:
    """Return a trimmed environment value, treating blank values as unset."""

    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def resolve_path(
    value: str | os.PathLike[str] | None,
    default: str | os.PathLike[str] | Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Resolve a path, interpreting relative values from the project root."""

    candidate = value if value is not None else default
    if candidate is None:
        raise ValueError("a path or default path is required")
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base_dir or globals().get("PROJECT_ROOT", PACKAGE_DIR.parents[1])
    return (Path(root) / path).resolve()


def _env_path(name: str, default: Path) -> Path:
    return resolve_path(env_value(name), default)


def _optional_env_path(name: str) -> Path | None:
    value = env_value(name)
    return resolve_path(value) if value else None


# For /<checkout>/scripts/RQ3/config.py this is /<checkout>.  The explicit
# override is useful when the scripts are copied into another checkout.
PROJECT_ROOT = _env_path("CUE_MEM_PROJECT_ROOT", PACKAGE_DIR.parents[1])
RQ3_ROOT = _env_path("CUE_MEM_RQ3_ROOT", PROJECT_ROOT / "RQ3")

DATA_DIR = _env_path("RQ3_DATA_DIR", RQ3_ROOT / "data")
HISTORY_DIR = _env_path("RQ3_HISTORY_DIR", DATA_DIR / "history_dialogue")
IMAGE_DIR = _env_path("RQ3_IMAGE_DIR", DATA_DIR / "event_image")
VOICE_DIR = _env_path("RQ3_VOICE_DIR", DATA_DIR / "voice_mixed_000_002")
QA_IMAGE_DIR = _env_path("RQ3_QA_IMAGE_DIR", DATA_DIR / "qa_image")
EMBEDDING_DIR = _env_path("RQ3_EMBEDDING_DIR", RQ3_ROOT / "embeddings")
RESULT_DIR = _env_path("RQ3_RESULT_DIR", RQ3_ROOT / "results")
PROMPT_DIR = _env_path("RQ3_PROMPT_DIR", RQ3_ROOT / "prompts")

ADVERSARIAL_QA_FILE = _env_path(
    "RQ3_ADVERSARIAL_QA_FILE",
    DATA_DIR / "qa_adversarial_llm_mcq_000_002.json",
)
ENTITY_SOURCE_DIR = _optional_env_path("RQ3_ENTITY_SOURCE_DIR")
CORRECT6_FILE = _optional_env_path("RQ3_CORRECT6_FILE")


def profile_paths(history_dir: Path | None = None) -> list[Path]:
    """Return the conventional profile files under ``history_dir``."""

    base = history_dir or HISTORY_DIR
    return [base / f"history_with_qa_p{p_id}.json" for p_id in range(3)]


PROFILE_FILES = profile_paths()


# Evaluation service configuration.  There are intentionally no URL or key
# literals here.  The generic names match the public repository's shared
# environment convention; RQ3-specific names take precedence where useful.
EVAL_MODEL_PROVIDER = env_value("RQ3_EVAL_MODEL_PROVIDER", "vllm")
OMNI_MODEL = env_value("RQ3_OMNI_MODEL", env_value("CUE_MEM_LLM_MODEL", "qwen3-omni"))
OMNI_API_BASE = env_value("RQ3_OMNI_API_BASE", env_value("CUE_MEM_LLM_BASE_URL"))
OMNI_API_KEY = env_value("RQ3_OMNI_API_KEY", env_value("CUE_MEM_LLM_API_KEY"))

ALIYUN_OMNI_MODEL = env_value("RQ3_ALIYUN_OMNI_MODEL", "qwen3.5-omni-plus")
ALIYUN_OMNI_API_BASE = env_value("RQ3_ALIYUN_OMNI_API_BASE")
ALIYUN_OMNI_API_KEY = env_value(
    "RQ3_ALIYUN_OMNI_API_KEY",
    env_value("DASHSCOPE_API_KEY"),
)


# Embedding service configuration.  These aliases preserve compatibility
# with existing local setups without placing a credential in source control.
EMBEDDING_PROVIDER = env_value("RQ3_EMBEDDING_PROVIDER", "imagebind")
GEMINI_EMBEDDING_MODEL = env_value(
    "RQ3_GEMINI_EMBEDDING_MODEL",
    env_value("GEMINI_EMBEDDING_MODEL", "google/gemini-embedding-2"),
)
GEMINI_EMBEDDING_API_BASE = env_value(
    "RQ3_GEMINI_EMBEDDING_API_BASE",
    env_value("GEMINI_EMBEDDING_API_BASE", env_value("OPENROUTER_API_BASE")),
)
GEMINI_EMBEDDING_API_KEY = env_value(
    "RQ3_GEMINI_EMBEDDING_API_KEY",
    env_value(
        "GEMINI_EMBEDDING_API_KEY",
        env_value("GEMINI_API_KEY", env_value("OPENROUTER_API_KEY")),
    ),
)
GEMINI_EMBEDDING_DIM = int(env_value("RQ3_GEMINI_EMBEDDING_DIM", "3072"))


# ImageBind is an optional external dependency.  Its checkpoint is not part
# of this public staging directory.
IMAGEBIND_DEVICE = env_value("RQ3_IMAGEBIND_DEVICE", "cuda:0")
IMAGEBIND_MODEL = env_value("RQ3_IMAGEBIND_MODEL", "imagebind_huge")
IMAGEBIND_EMBEDDING_DIM = int(env_value("RQ3_IMAGEBIND_EMBEDDING_DIM", "1024"))


DEFAULT_TOP_K = 5
RETRIEVAL_TOP_K_LIST = [1, 3, 5, 10]
VARIANTS = ["TT", "TM", "MT", "MM"]


_URL_RE = re.compile(r"(?i)\b(?:https?|wss?)://[^\s'\"<>]+")


def redact_sensitive_text(value: object, *secrets: str | None) -> str:
    """Redact runtime credentials and URLs before logging or persisting text."""

    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    return _URL_RE.sub("<redacted-url>", text)


def required_runtime_value(
    value: str | None,
    *,
    argument_name: str,
    env_names: Iterable[str],
) -> str:
    """Validate a runtime credential/endpoint without echoing its value."""

    if value and value.strip():
        return value.strip()
    names = ", ".join(env_names)
    raise RuntimeError(
        f"missing runtime value for {argument_name}; "
        f"pass the command-line option or set one of: {names}"
    )
