"""Shim de compatibilidade — use ``pepmem.peptide_utils``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pepmem.peptide_utils import *  # noqa: E402, F403
