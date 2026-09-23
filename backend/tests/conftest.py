"""Shared pytest setup: make ``app`` importable when running ``pytest``
straight from the backend directory (no package install required).
"""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
