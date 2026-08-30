"""Public RQ1/RQ2 benchmark package.

The original benchmark was executed from ``benchmark/run`` and therefore
relied on the current working directory for imports.  Keep that layout
working while also allowing package execution from a repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
BENCHMARK_PACKAGE_ROOT = PACKAGE_ROOT / "benchmark"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(BENCHMARK_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_PACKAGE_ROOT))
