"""Stable paths independent of a test module's directory depth."""
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
