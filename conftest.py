"""Root conftest — ensure the repo root is importable so `from src import ...` and
`from tests import fixtures` work under pytest regardless of invocation directory."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
