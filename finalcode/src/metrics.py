"""Macro F0.5 exactly as described in the challenge README."""
import numpy as np
import pandas as pd


def load_gt(path):
    """Load a ground-truth file (parquet or TSV) as {s1_id: set(matched ids)}."""
    gt = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t", dtype=str,
                                                                             keep_default_na=False)
    return {s: set(x for x in m.split(",") if x) for s, m in zip(gt.source1_entity_id, gt.matched_entity_ids)}


def f05(pred, true):
    """F0.5 of one S1 entity: 1 for a correctly empty prediction of a singleton, else 1.25*P*R/(0.25*P+R)."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, gt, ids=None):
    """pred: dict s1 -> set; gt: dict s1 -> set; ids: S1 ids to average over."""
    ids = list(gt.keys()) if ids is None else ids
    return float(np.mean([f05(pred.get(s, set()), gt[s]) for s in ids]))


def pairs_to_dict(s1, qid):
    """Group (s1, query) pairs into {s1: set of query ids}."""
    out = {}
    for a, b in zip(s1, qid):
        out.setdefault(a, set()).add(b)
    return out
