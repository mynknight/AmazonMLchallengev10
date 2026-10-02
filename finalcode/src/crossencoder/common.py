"""Paths and helpers shared by the cross-encoder scripts.

The cross-encoder method runs on top of the v2 pipeline (blocking, 64 hand features, LightGBM) in
business_entity_resolution/src and its run directory.  Nothing there is modified; this folder only
reads it.  Every path can be overridden with an environment variable (defaults = this server).
"""
import os
import sys

HOME = os.environ.get("ER_ROOT", os.path.expanduser("~/MLchallenge"))
_HERE = os.path.dirname(os.path.abspath(__file__))
# submission package layout: src/crossencoder/ sits inside the v2 src/ folder
_PARENT = os.path.dirname(_HERE)
_DEFAULT_SRC = _PARENT if os.path.exists(os.path.join(_PARENT, "pairs.py")) else \
    os.path.join(HOME, "business_entity_resolution", "src")
OLD_SRC = os.environ.get("ER_V2_SRC", _DEFAULT_SRC)
V2 = os.environ.get("ER_V2_WORK", os.path.join(HOME, "work_v2"))     # v2 run: features, models, scores
RAW = os.environ.get("ER_RAW", V2)            # raw TSV -> parquet copies (entity_id, name, address, country)
WORK = os.environ.get("ER_CE_WORK", os.path.join(HOME, "crossencoder", "work"))
DATA = os.environ.get("ER_DATA", os.path.join(HOME, "6ab10eb3b23ba_student_resource", "student_resource", "dataset"))

if OLD_SRC not in sys.path:
    sys.path.insert(0, OLD_SRC)

THREADS = int(os.environ.get("ER_THREADS", 24))

# the v2 LightGBM (stage 1) is used as a filter: pairs below this probability are dropped before
# the cross-encoder (on validation this keeps all but 4 of 452k true pairs)
P1_MIN = float(os.environ.get("CE_P1_MIN", 1e-4))
