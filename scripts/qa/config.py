"""Canonical paths shared by QA scripts.

The scripts live under ``scripts/qa`` but the generated datasets are kept in
the project-level ``qa`` directory.  The benchmark directory is configurable
so this code works both before and after the Mem-Gallery -> RQ1_RQ2 rename.
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.common.paths import project_path, resolve_path


QA_ROOT = project_path("qa")
PROFILE_ROOT = project_path("profile")


def _benchmark_root() -> Path:
    configured = os.environ.get("CUE_MEM_BENCHMARK_ROOT", "").strip()
    if configured:
        return resolve_path(configured)

    renamed = project_path("RQ1_RQ2", "benchmark")
    legacy = project_path("Mem-Gallery-main", "benchmark")
    if renamed.exists() or not legacy.exists():
        return renamed
    return legacy


BENCHMARK_ROOT = _benchmark_root()
BENCHMARK_RUN_ROOT = BENCHMARK_ROOT / "run"
BENCHMARK_DATA_ROOT = BENCHMARK_ROOT / "data"


def qa_path(*parts: str | os.PathLike[str]) -> Path:
    return QA_ROOT.joinpath(*parts)


def profile_path(*parts: str | os.PathLike[str]) -> Path:
    return PROFILE_ROOT.joinpath(*parts)


def benchmark_path(*parts: str | os.PathLike[str]) -> Path:
    return BENCHMARK_ROOT.joinpath(*parts)
