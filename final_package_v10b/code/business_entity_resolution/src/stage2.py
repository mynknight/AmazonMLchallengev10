"""Second-stage model (HANDOFF §5.4/§5.5): re-score pairs with context built from the stage-1
probabilities of the query's other candidates and of the S1 entity's other candidate queries.

Stage 1 is trained on the training states, so its probabilities on the validation states are
out-of-sample; stage 2 is therefore trained on the validation pairs.  Its own quality is
measured by 2-fold cross-fitting over S1 entities (train on one half of the validation S1
entities, predict the other half, and vice versa).

    python stage2.py <work> <stage1_model_name> [eval|fit]
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from decide import tune_ef, tune_fine
from metrics import load_gt
from pipeline import _hash01

THREADS = int(os.environ.get("ER_THREADS", os.cpu_count()))
PARAMS2 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               num_threads=THREADS, verbose=-1, seed=1)
ROUNDS2 = 600


def _rank_desc(codes, v):
    order = np.lexsort((-v, codes))
    cs = codes[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    idx = np.arange(len(cs))
    r = np.empty(len(cs), np.float32)
    r[order] = idx - np.maximum.accumulate(np.where(start, idx, 0)) + 1
    return r


def _top2(codes, v):
    order = np.lexsort((-v, codes))
    cs, vs = codes[order], v[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    ng = codes.max() + 1
    b1 = np.full(ng, 0.0, np.float32); b2 = np.full(ng, 0.0, np.float32)
    b1[cs[first]] = vs[first]
    sec = np.nonzero(~first & np.r_[False, first[:-1]])[0]
    b2[cs[sec]] = vs[sec]
    return b1[codes], b2[codes]


def s2_features(sc):
    """sc: qid, s1, prob (one row per candidate pair).  Returns a float32 feature frame."""
    p = sc.prob.values.astype(np.float32)
    qc = pd.factorize(sc.qid.values)[0]
    dc = pd.factorize(sc.s1.values)[0]
    f = {"p": p, "logit": np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))}
    # query side: the query belongs to at most one S1
    q1, q2 = _top2(qc, p)
    f["q_rank"] = _rank_desc(qc, p)
    f["q_best"] = q1
    f["q_other_best"] = np.where(p >= q1, q2, q1)          # best competing S1 for this query
    f["q_margin"] = p - f["q_other_best"]
    f["q_sum_other"] = np.bincount(qc, weights=p)[qc] - p
    f["q_ncand"] = np.bincount(qc)[qc].astype(np.float32)
    # the query's assigned S1 = its best candidate; per S1, what else is assigned to it
    is_best = (f["q_rank"] == 1)
    f["is_best"] = is_best.astype(np.float32)
    pb = np.where(is_best, p, 0.0)
    for t in (0.5, 0.8, 0.95):
        n = np.bincount(dc, weights=(pb >= t).astype(np.float64))[dc]
        f[f"d_n_assigned_{t}"] = (n - ((pb >= t) & is_best)).astype(np.float32)   # excluding this row
    f["d_sum_assigned_other"] = (np.bincount(dc, weights=pb)[dc] - pb).astype(np.float32)
    d1, d2 = _top2(dc, p)
    f["d_rank"] = _rank_desc(dc, p)
    f["d_other_best"] = np.where(p >= d1, d2, d1)
    f["d_sum_other"] = (np.bincount(dc, weights=p)[dc] - p).astype(np.float32)
    f["d_ncand"] = np.bincount(dc)[dc].astype(np.float32)
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def consistency_features(sc, feats, rec):
    """HANDOFF §5.5: similarity of the query to the OTHER strong queries of the same S1 (queries whose
    best candidate is this S1 with p >= 0.8).  rec: normalised records indexed by entity_id.
    Test-like copies ("<id>~k") are mapped back to their base record and never compared with
    their own twin."""
    from rapidfuzz import fuzz, process
    base = pd.Series(sc.qid.values).str.replace(r"~\d+$", "", regex=True).values
    strong = (feats.is_best.values == 1) & (feats.p.values >= 0.8)
    st = pd.DataFrame({"s1": sc.s1.values[strong], "q2": base[strong]})
    rows = pd.DataFrame({"row": np.arange(len(sc)), "s1": sc.s1.values, "q": base})
    m = rows.merge(st, on="s1")
    m = m[m.q != m.q2]
    core = rec.core.reindex(m.q).values, rec.core.reindex(m.q2).values
    ad = rec.ad.reindex(m.q).values, rec.ad.reindex(m.q2).values
    nums = rec.nums.reindex(m.q).values, rec.nums.reindex(m.q2).values
    cp = lambda a, b, sc_: process.cpdist(list(a), list(b), scorer=sc_, workers=THREADS, dtype=np.float32)
    m = m.assign(n=cp(*core, fuzz.token_set_ratio), a=cp(*ad, fuzz.token_set_ratio),
                 u=cp(*nums, fuzz.token_set_ratio))
    agg = m.groupby("row").agg(c_name=("n", "max"), c_addr=("a", "max"), c_nums=("u", "max"),
                               c_n=("n", "size"))
    out = pd.DataFrame(index=np.arange(len(sc)))
    for c in ("c_name", "c_addr", "c_nums", "c_n"):
        out[c] = agg[c].reindex(out.index).fillna(-1 if c != "c_n" else 0).values.astype(np.float32)
    return out


def _folds(s1, n_folds):
    u = pd.unique(s1)
    return pd.Series((_hash01(u) * n_folds).astype(int), index=u).reindex(s1).values


def crossfit(train_sc, train_feats, apply_sc, apply_feats, n_folds=2):
    """Out-of-fold stage-2 probabilities for apply_sc: for each S1-entity fold, a model trained on
    the train_sc rows of the OTHER folds scores the apply_sc rows of this fold.  Training on the
    plain validation set keeps the model blind to the duplicated distractors of the test-like set
    (exact twins would otherwise be an exploitable artefact)."""
    ftr, fap = _folds(train_sc.s1.values, n_folds), _folds(apply_sc.s1.values, n_folds)
    out = np.zeros(len(apply_sc), np.float32)
    for k in range(n_folds):
        tr = ftr != k
        b = lgb.train(PARAMS2, lgb.Dataset(train_feats.values[tr], train_sc.label.values[tr],
                                           feature_name=list(train_feats.columns)), ROUNDS2)
        out[fap == k] = b.predict(apply_feats.values[fap == k], num_threads=THREADS)
    return out


def main(work, name, mode="eval"):
    from train import testlike, val_entities
    sc = pd.read_parquet(os.path.join(work, f"val_scores_{name}.parquet"))
    ents = val_entities(work)
    gt = load_gt(os.path.join(work, "train_gt.parquet"))
    ids = ents.entity_id.tolist()
    gsub = {s: gt[s] for s in ids}
    res = {}
    sc = sc.reset_index(drop=True)
    cons = os.environ.get("ER_S2_CONS", "0") == "1"
    rec = None
    if cons:
        rec = pd.concat([pd.read_parquet(os.path.join(work, f"train_s{s}_norm.parquet"),
                                         columns=["entity_id", "core", "ad", "nums"]) for s in (2, 3)]).set_index("entity_id")
    feats_std = s2_features(sc)
    if cons:
        feats_std = pd.concat([feats_std, consistency_features(sc, feats_std, rec)], axis=1)
    for tag, data in (("std", sc), ("test-like", testlike(sc, work))):
        data = data.reset_index(drop=True)
        feats = s2_features(data)
        if cons:
            feats = pd.concat([feats, consistency_features(data, feats, rec)], axis=1)
        data = data.assign(prob=crossfit(sc, feats_std, data, feats))
        t, f = tune_fine(data, gsub, ids)
        (mi, fl), fe = tune_ef(data, gsub, ids)
        t1, f1 = tune_fine(data.assign(prob=feats.p.values), gsub, ids)
        res[tag] = dict(threshold=t, f05=f, ef_miss=mi, ef_floor=fl, ef_f05=fe, stage1_f05=f1)
        print(f"[{tag}] stage-1 F0.5 {f1:.4f} | stage-2 threshold {t:.2f} -> F0.5 {f:.4f} | "
              f"stage-2 expected-F ({mi}, {fl}) -> {fe:.4f}", flush=True)
    with open(os.path.join(work, f"{name}_stage2_eval.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
