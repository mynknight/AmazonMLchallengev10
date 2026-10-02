"""Paths for stage D (France veto + CE-confirmed additions, "V10"). Reads stage C (src/v9) and the
stage-B v6a France scores; writes into ER_V10_WORK (default <ER_V2_WORK>/v10)."""
import os
import sys

V9_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v9")
if V9_SRC not in sys.path:
    sys.path.insert(0, V9_SRC)
from common9 import CE_WORK, DATA, V2, WORK9  # noqa: E402,F401  (also puts the v2 / CE sources on sys.path)

WORK10 = os.environ.get("ER_V10_WORK", os.path.join(V2, "v10"))
CE4 = os.path.join(WORK9, "ce4")
os.makedirs(WORK10, exist_ok=True)
