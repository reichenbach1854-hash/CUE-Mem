"""Project-relative path helpers.

All public scripts resolve relative paths from the repository root instead of
from the caller's current working directory.  Set ``CUE_MEM_PROJECT_ROOT``
when the scripts are used from a copied checkout.
"""

from __future__ import annotations

import os
from pathlib import Path


_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_configured_root = os.environ.get("CUE_MEM_PROJECT_ROOT", "").strip()
PROJECT_ROOT = Path(_configured_root or _DEFAULT_ROOT).expanduser().resolve()


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a path below the project root."""

    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(value: str | os.PathLike[str] | None, default: Path | None = None) -> Path:
    """Resolve a CLI path relative to the project root.

    Absolute paths are preserved.  ``None`` uses ``default`` and raises a
    useful error if neither value was supplied.
    """

    if value is None:
        if default is None:
            raise ValueError("a path or default path is required")
        candidate = default
    else:
        candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def project_relative(path: str | os.PathLike[str] | Path) -> str:
    """Return a POSIX path relative to the project root when possible."""

    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()
