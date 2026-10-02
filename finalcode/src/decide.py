"""Turn pair probabilities into per-entity match sets.

Step 1 (assignment): every S2/S3 record keeps only its highest-probability S1
candidate — a record belongs to at most one business.

Step 2 (selection), two variants:
  * threshold   : keep assigned records with prob >= t.
  * expected F  : per S1 entity, sort its assigned records by prob and choose the
                  prefix size m maximising the expected F0.5,
                      E[F](m) ~= 1.25 * sum_{j<=m} p_j / (0.25 * N + m),   m >= 1
                      E[F](0)  = prod_j (1 - p_j)            (entity is a singleton)
                  where N = sum_j p_j + miss (expected number of true matches).
                  Records below a floor probability are never selected.
"""
import numpy as np
import pandas as pd

from metrics import macro_f05


def apply_rules(c, prob):
    """Deterministic data rules applied to model probabilities before any decision.

    Unknown native-script words (x_nat_unk_q > 0): native words missing from the transliteration
    dictionary form a closed set of 171 business-type words (stores, motors, bakery, ... in nine
    scripts) that the data generator adds to create sibling distractors.  In training all 64,020
    records carrying one are distractors, and the test set uses exactly the same 171 words, so
    such a query is never matched."""
    prob = np.asarray(prob, np.float64).copy()
    if "x_nat_unk_q" in c:
        prob[np.asarray(c["x_nat_unk_q"]) > 0] = 0.0
    return prob


def best_per_query(sc):
    """Keep, for every query, the row with the highest probability."""
    sc = sc.sort_values(["qid", "prob"], ascending=[True, False])
    return sc.drop_duplicates("qid", keep="first")


def _to_dict(s1, q):
    """Group (s1, query) pairs into {s1: set of matched query ids}."""
    out = {}
    for a, b in zip(s1, q):
        out.setdefault(a, set()).add(b)
    return out


def decide(sc, t, best=None):
    """Threshold rule: keep each query's best S1 candidate if its probability >= t; return {s1: set(query ids)}."""
    b = best if best is not None else best_per_query(sc)
    b = b[b.prob >= t]
    return _to_dict(b.s1.values, b.qid.values)


def expected_f_select(s1, q, p, miss=0.0, floor=0.05, beta2=0.25):
    """s1/q/p: assigned pairs (one row per query). Returns boolean mask of selected rows."""
    order = np.lexsort((-p, s1))
    s1o, po = s1[order], p[order]
    sel = np.zeros(len(p), bool)
    bounds = np.r_[0, np.nonzero(s1o[1:] != s1o[:-1])[0] + 1, len(s1o)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        pp = po[a:b]
        pp = pp[pp >= floor]
        if len(pp) == 0:
            continue
        n_true = pp.sum() + miss
        tp = np.cumsum(pp)
        m = np.arange(1, len(pp) + 1)
        ef = (1 + beta2) * tp / (beta2 * n_true + m)
        e0 = np.prod(1 - pp)
        k = int(np.argmax(ef))
        if ef[k] > e0:
            sel[order[a:a + k + 1]] = True
    return sel


def decide_ef(sc, best=None, **kw):
    """Expected-F0.5 rule: per S1 entity, keep the best-per-query candidates that maximise the expected F0.5."""
    b = best if best is not None else best_per_query(sc)
    m = expected_f_select(b.s1.values, b.qid.values, b.prob.values.astype(np.float64), **kw)
    return _to_dict(b.s1.values[m], b.qid.values[m])


def tune(sc, gt, ids, grid=None):
    """Coarse threshold search (0.20-0.95, step 0.05) maximising exact macro F0.5; returns (threshold, score)."""
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.96, 0.05), 3)
    best = best_per_query(sc)
    scores = [(t, macro_f05(decide(sc, t, best), gt, ids)) for t in grid]
    return max(scores, key=lambda x: x[1])


def threshold_curve(sc, gt, ids, grid):
    """Exact macro F0.5 of the threshold rule for every t in grid, vectorised (same result as
    macro_f05(decide(sc, t), gt, ids)).  sc needs qid, s1, prob, label."""
    b = best_per_query(sc)
    code = pd.Index(ids).get_indexer(b.s1.values)
    keep = code >= 0
    code, prob, tp = code[keep], b.prob.values[keep], b.label.values[keep].astype(np.float64)
    n_true = np.array([len(gt[s]) for s in ids], np.float64)
    out = []
    for t in grid:
        m = prob >= t
        npred = np.bincount(code[m], minlength=len(ids))
        ntp = np.bincount(code[m], weights=tp[m], minlength=len(ids))
        f = np.where(n_true == 0, (npred == 0).astype(np.float64),
                     1.25 * ntp / np.maximum(0.25 * n_true + npred, 1e-12))
        out.append(float(f.mean()))
    return np.asarray(out)


def tune_fine(sc, gt, ids, grid=None):
    """Fine threshold search (0.30-0.99, step 0.01) on the fast threshold curve; returns (threshold, score)."""
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.991, 0.01), 3)
    f = threshold_curve(sc, gt, ids, grid)
    k = int(np.argmax(f))
    return float(grid[k]), float(f[k])


def tune_ef(sc, gt, ids, misses=(0.0, 0.1, 0.25, 0.5), floors=(0.05, 0.2, 0.35)):
    """Grid-search the miss rate and probability floor of the expected-F0.5 rule; returns ((miss, floor), score)."""
    best = best_per_query(sc)
    res = [((mi, fl), macro_f05(decide_ef(sc, best, miss=mi, floor=fl), gt, ids))
           for mi in misses for fl in floors]
    return max(res, key=lambda x: x[1])
