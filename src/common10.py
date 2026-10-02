"""Paths for V10. V10 builds on V9 (~/MLchallenge_v9) and the parallel session's v6a France model
(~/MLchallenge/crossencoder); it only reads them and writes into ~/MLchallenge_v10."""
import os
import sys

V9_SRC = "/home/mayankc/MLchallenge_v9/src"
if V9_SRC not in sys.path:
    sys.path.insert(0, V9_SRC)
from common9 import CE_WORK, DATA, V2, WORK9  # noqa: E402,F401  (also puts the v2 / CE sources on sys.path)

V10 = "/home/mayankc/MLchallenge_v10"
WORK10 = os.path.join(V10, "work")
CE4 = os.path.join(WORK9, "ce4")
