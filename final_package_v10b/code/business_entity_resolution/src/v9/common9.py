"""Paths shared by the stage-C scripts (US/India final matcher, "V9").

Layout (defaults; every path can be overridden with an environment variable, run_all.sh sets them):
    ER_V2_WORK        stage-A work dir (v2: normalised records, features, stage-1 models)
    ER_CE_STAGE_WORK  stage-B work dir (2-fold cross-encoder, pair lists, v6a test scores)
    ER_V9_WORK        this stage's work dir (4-fold cross-encoders in ce4/, deeper retrieval in ea/)
    ER_DATA           dataset dir with train/ and test/
Note: the cross-encoder helpers read ER_CE_WORK; for this stage it must point at <ER_V9_WORK>/ce4.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_HERE))                  # business_entity_resolution/
_ROOT = os.path.dirname(_PKG)
V2 = os.environ.get("ER_V2_WORK", os.path.join(_ROOT, "work"))
CE_WORK = os.environ.get("ER_CE_STAGE_WORK", os.path.join(V2, "ce"))
WORK9 = os.environ.get("ER_V9_WORK", os.path.join(V2, "v9"))
DATA = os.environ.get("ER_DATA", os.path.join(_ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset"))
CE_SRC = os.path.join(os.path.dirname(_HERE), "crossencoder")
OLD_SRC = os.path.dirname(_HERE)                                 # src/ (v2 code)
for p in (CE_SRC, OLD_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ER_THREADS", "48")
THREADS = int(os.environ["ER_THREADS"])
