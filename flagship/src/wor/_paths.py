"""Path setup: make the vendored ``jlens`` (flagship/third_party/jlens) importable
without pip. Imported for its side effect by ``wor/__init__.py``."""

from __future__ import annotations

import sys
from pathlib import Path

FLAGSHIP_DIR = Path(__file__).resolve().parents[2]
THIRD_PARTY_DIR = FLAGSHIP_DIR / "third_party"
DATA_DIR = FLAGSHIP_DIR / "data"

if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))
