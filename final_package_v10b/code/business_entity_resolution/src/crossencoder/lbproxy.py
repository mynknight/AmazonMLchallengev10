"""Label-free estimate of macro F0.5: sample the truth from the model's own probabilities.

    python lbproxy.py [--draws 8] [--shift 1.0]

For every query the model gives P(query matches candidate j) = p_j (renormalised when the sum
exceeds 1) and P(no match) = the rest.  One Monte-Carlo draw samples one outcome per query, the
submission rule (best candidate per query, per-country threshold) gives the prediction, and the
exact per-entity F0.5 is averaged.  Run on validation it can be compared with the real F0.5 (a
calibration check); run on the test candidates it estimates the leaderboard score without labels,
e.g. whether the test set is harder than the validation states.

--shift multiplies the match odds of every query by a constant before sampling (prior shift:
test has ~40% distractor records vs 26% in train, i.e. odds x (60/40)/(74/26) = 0.53 if the model
is calibrated on train).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import V2, WORK


def expected_f(q, s1, prob, thr, ents, draws=8, shift=1.0, miss_rate=0.0135, seed=0, sample_prob=None):
    """q, s1: int codes / ids per pair; prob: decision probability; thr: per-pair threshold;
    ents: array of S1 ids of the evaluated universe; sample_prob: probability used to sample the
    truth (default prob)."""
    rng = np.random.default_rng(seed)
    sp = prob if sample_prob is None else sample_prob
    if shift != 1.0:
        o = sp / np.maximum(1 - sp, 1e-9) * shift
        sp = o / (1 + o)
    qc, qu = pd.factorize(q)
    ec = pd.Index(ents).get_indexer(s1)
    keep = ec >= 0
    qc, ec, prob, sp, thr = qc[keep], ec[keep], prob[keep], sp[keep], thr[keep]
    # prediction: best candidate per query, kept if >= threshold
    order = np.lexsort((-prob, qc))
    first = order[np.r_[True, qc[order][1:] != qc[order][:-1]]]
    pred = first[prob[first] >= thr[first]]
    npred = np.bincount(ec[pred], minlength=len(ents)).astype(np.float64)
    # sampling: per query, cumulative probabilities in (arbitrary) candidate order
    tot = np.bincount(qc, weights=sp)
    norm = np.maximum(tot, 1.0)[qc]
    pn = sp / norm
    o2 = np.argsort(qc, kind="stable")
    qs, ps = qc[o2], pn[o2]
    cum = np.cumsum(ps)
    start = np.r_[0, np.nonzero(qs[1:] != qs[:-1])[0] + 1]
    base = np.repeat(cum[start] - ps[start], np.diff(np.r_[start, len(qs)]))
    hi = cum - base          # upper edge of each candidate's interval within its query
    lo = hi - ps
    is_pred = np.zeros(len(qc), bool)
    is_pred[pred] = True
    is_pred_o = is_pred[o2]
    ec_o = ec[o2]
    out = []
    for _ in range(draws):
        u = rng.random(len(qu))[qs]
        hit = (u >= lo) & (u < hi)            # this candidate is the query's true S1
        ntrue = np.bincount(ec_o[hit], minlength=len(ents)).astype(np.float64)
        tp = np.bincount(ec_o[hit & is_pred_o], minlength=len(ents)).astype(np.float64)
        ntrue_all = ntrue * (1 + miss_rate / (1 - miss_rate))
        f = np.where(ntrue == 0, (npred == 0).astype(np.float64),
                     1.25 * tp / np.maximum(0.25 * ntrue_all + npred, 1e-12))
        out.append(f)
    return np.mean(out, axis=0)


def load_thr(path):
    with open(path) as fh:
        return json.load(fh)


def run(draws=8, shift=1.0):
    from train import val_entities
    thr_main = load_thr(os.path.join(V2, "lgb_v2_thresholds.json"))
    thr_fr = load_thr(os.path.join(V2, "lgb_v2_novocab_thresholds.json"))
    # ---- validation: model estimate vs truth
    va = pd.read_parquet(os.path.join(V2, "val_scores_lgb_v2.parquet"))
    ents = val_entities(V2)
    for c in ("US", "India"):
        e = ents[ents.country == c].entity_id.values
        m = (va.country == c).values
        t = np.full(m.sum(), thr_main[c]["threshold"])
        f = expected_f(va.qid.values[m], va.s1.values[m], va.prob.values[m], t, e, draws, shift)
        print(f"val {c}: estimated F0.5 {f.mean():.4f} (actual std {thr_main['std'][c]['f05']:.4f}, "
              f"test-like {thr_main['testlike'][c]['f05']:.4f}) entities {len(e):,}", flush=True)
    # ---- test
    import pyarrow.dataset as ds
    te = ds.dataset(os.path.join(V2, "test_scores.parquet")).to_table(
        filter=ds.field("prob") >= 1e-5).to_pandas(strings_to_categorical=True)
    s1 = pd.read_parquet(os.path.join(V2, "test_s1.parquet"), columns=["entity_id", "country"])
    s1["id"] = s1.entity_id.str.slice(3).astype(np.int64)
    res = {}
    for c in ("US", "India", "France"):
        e = s1[s1.country == c].id.values
        m = (te.country == c).values
        th = (thr_fr["global"] if c == "France" else thr_main[c])["threshold"]
        f = expected_f(te.q.values[m], te.s1.values[m], te.prob.values[m], np.full(m.sum(), th), e, draws, shift)
        res[c] = (f.mean(), len(e))
        print(f"test {c}: estimated F0.5 {f.mean():.4f} entities {len(e):,}", flush=True)
    n = sum(v[1] for v in res.values())
    print(f"test all: estimated F0.5 {sum(v[0] * v[1] for v in res.values()) / n:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--shift", type=float, default=1.0)
    a = ap.parse_args()
    run(a.draws, a.shift)
