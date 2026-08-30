"""Runtime paths for the RQ1/RQ2 benchmark.

Only code and small prompt templates are shipped with this package.  The
benchmark data, generated results, and caches remain outside the package and
are selected at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.common.paths import PROJECT_ROOT as REPOSITORY_ROOT
from scripts.common.paths import resolve_path


PACKAGE_BENCHMARK_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _benchmark_root() -> Path:
    configured = os.environ.get("CUE_MEM_BENCHMARK_ROOT", "").strip()
    if configured:
        return resolve_path(configured)

    configured_project = os.environ.get("CUE_MEM_PROJECT_ROOT", "").strip()
    project_root = resolve_path(configured_project) if configured_project else REPOSITORY_ROOT

    # The public checkout keeps code under scripts/RQ1_RQ2, while benchmark
    # data/results are intentionally external to that package.  Prefer the
    # renamed top-level data root, then the legacy source-tree name, and only
    # If no external data has been checked out yet, use the renamed top-level
    # root anyway so generated files do not land beside the source package.
    for candidate in (
        project_root / "RQ1_RQ2" / "benchmark",
        project_root / "Mem-Gallery-main" / "benchmark",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (project_root / "RQ1_RQ2" / "benchmark").resolve()


BENCHMARK_ROOT = _benchmark_root()
DATA_ROOT = _path_from_env("CUE_MEM_DATA_ROOT", BENCHMARK_ROOT / "data")
RESULT_ROOT = _path_from_env("CUE_MEM_RESULT_ROOT", BENCHMARK_ROOT / "result_debug")
QUESTION_ONLY_RESULT_ROOT = _path_from_env(
    "CUE_MEM_QUESTION_ONLY_RESULT_ROOT", BENCHMARK_ROOT / "result_question_only"
)
ORACLE_RESULT_ROOT = _path_from_env(
    "CUE_MEM_ORACLE_RESULT_ROOT", BENCHMARK_ROOT / "result_oracle_evidence"
)
JUDGE_RESULT_ROOT = _path_from_env(
    "CUE_MEM_JUDGE_RESULT_ROOT", BENCHMARK_ROOT / "result_debug_llm_as_judge"
)
TRIMMED_RESULT_ROOT = _path_from_env(
    "CUE_MEM_TRIMMED_RESULT_ROOT", BENCHMARK_ROOT / "result_debug_trimmed"
)
PROMPT_ROOT = _path_from_env(
    "CUE_MEM_PROMPT_ROOT", PACKAGE_BENCHMARK_ROOT / "prompt"
)
CACHE_ROOT = _path_from_env("CUE_MEM_CACHE_ROOT", BENCHMARK_ROOT / ".cache")
MEMORY_CACHE_ROOT = _path_from_env(
    "CUE_MEM_MEMORY_CACHE_ROOT", BENCHMARK_ROOT / ".memory_cache"
)
QA_ROOT = _path_from_env("CUE_MEM_QA_ROOT", REPOSITORY_ROOT / "qa")
EVENT_ROOT = _path_from_env("CUE_MEM_EVENT_ROOT", REPOSITORY_ROOT / "event")
RQ3_ROOT = _path_from_env("CUE_MEM_RQ3_ROOT", REPOSITORY_ROOT / "RQ3")

DIALOG_ROOT = DATA_ROOT / "dialog"
IMAGE_ROOT = DATA_ROOT / "image"
VOICE_ROOT = DATA_ROOT / "voice"


def benchmark_path(*parts: str | os.PathLike[str]) -> Path:
    """Resolve a path below the externally configurable benchmark root."""

    return BENCHMARK_ROOT.joinpath(*parts)


def data_path(*parts: str | os.PathLike[str]) -> Path:
    return DATA_ROOT.joinpath(*parts)


def result_path(*parts: str | os.PathLike[str]) -> Path:
    return RESULT_ROOT.joinpath(*parts)


def cache_path(*parts: str | os.PathLike[str]) -> Path:
    return CACHE_ROOT.joinpath(*parts)


def memory_cache_path(*parts: str | os.PathLike[str]) -> Path:
    return MEMORY_CACHE_ROOT.joinpath(*parts)


def resolve_runtime_path(
    value: str | os.PathLike[str], *, root: Path = BENCHMARK_ROOT
) -> Path:
    """Resolve an absolute path as-is and a relative path below ``root``."""

    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
