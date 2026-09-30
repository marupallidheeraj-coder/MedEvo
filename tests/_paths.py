"""Puts backend/, research/ and data/ on sys.path so tests can import them by module name."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("backend", "research", "data"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
DATA_DIR = ROOT / "data"
