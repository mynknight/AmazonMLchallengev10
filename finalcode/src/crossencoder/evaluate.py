"""Validation of the cross-encoder signal on the v2 validation set (US {WA, IA, CT}, India {KL, OD}).

    python evaluate.py <ce_name>[+<ce_name>...] [--p1 v2|novocab]

Compares, with the same decision rule as the submission (best S1 per query, per-country threshold
tuned on the evaluated set, standard and test-like = distractor queries x1.9):
  v2        : stage-1 probability alone (the current submission)
  ce        : cross-encoder alone
  blend-lr  : logistic regression on [logit p1, ce], fitted leave-one-state-out
  stage2    : small LightGBM on p1 / ce and their within-query / within-S1 context,
              fitted leave-one-state-out (every entity is scored by a model that never saw its state)
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import THREADS, V2, WORK
from metrics import load_gt
from train import _tune_all, testlike, val_entities


def load_val(names, p1="v2"):
    """Validation pairs with the stage-1 probability (v2 or vocabulary-free model), mean CE score and the S1 state."""
    va = pd.read_parquet(os.path.join(WORK, "pairs_val.parquet"))
    if p1 == "novocab":  # the France model's stage-1 probability on the same rows
        nv = pd.read_parquet(os.path.join(V2, "val_scores_lgb_v2_novocab.parquet"), columns=["prob"])
        va["p1"] = nv.prob.values[va.row.values]
    ces = [np.load(os.path.join(WORK, f"ce_{n}_val.npy")) for n in names]
    va["ce"] = np.mean(ces, axis=0)
    s1n = pd.read_parquet(os.path.join(V2, "train_s1_norm.parquet"), columns=["entity_id", "state"])
    va["st"] = s1n.set_index("entity_id").state.reindex(va.s1.values).values
    return va


def logit(p, eps=1e-6):
    """Log-odds of probabilities, clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def context(df, col, key, prefix, missing=-20.0):
    """Within each group of `key`: rank of the row (0 = best), gap to the best *other* row, size."""
    codes = pd.factorize(df[key].values)[0]
    v = df[col].values.astype(np.float64)
    order = np.lexsort((-v, codes))
    c_o, v_o = codes[order], v[order]
    n = len(c_o)
    start = np.r_[True, c_o[1:] != c_o[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    rank_o = np.arange(n) - first
    nxt = np.minimum(first + 1, n - 1)
    second_o = np.where((first + 1 < n) & (c_o[nxt] == c_o), v_o[nxt], missing)
    other_o = np.where(rank_o == 0, second_o, v_o[first])
    size_o = np.bincount(codes)[c_o]
    for name, vals in (("rk", rank_o), ("gap", v_o - other_o), ("n", size_o)):
        out = np.empty(n, np.float32)
        out[order] = vals
        df[f"{prefix}_{name}"] = out
    return df


def stage2_features(va):
    """Add logit(p1) and query/S1 context features (rank, gap) of p1 and the CE score."""
    va["lp1"] = logit(va.p1.values)
    va = context(va, "lp1", "qid", "q_p1")
    va = context(va, "ce", "qid", "q_ce")
    va = context(va, "lp1", "s1", "d_p1")
    va = context(va, "ce", "s1", "d_ce")
    return va


S2_FEATS = ["lp1", "ce", "q_p1_rk", "q_p1_gap", "q_ce_rk", "q_ce_gap", "q_ce_n",
            "d_p1_rk", "d_p1_gap", "d_ce_rk", "d_ce_gap", "d_ce_n"]


def cv_predict(va, fit_predict):
    """Leave-one-S1-state-out predictions on validation with a fit_predict(train, test) function."""
    out = np.zeros(len(va))
    for st in sorted(va.st.dropna().unique()):
        te = (va.st == st).values
        out[te] = fit_predict(va[~te], va[te])
    return out


def fit_lr(tr, te):
    """Logistic regression on [logit(p1), CE score]; returns test probabilities."""
    from sklearn.linear_model import LogisticRegression
    X = lambda d: np.c_[logit(d.p1.values), d.ce.values]
    m = LogisticRegression(C=10.0, max_iter=500).fit(X(tr), tr.label.values)
    return m.predict_proba(X(te))[:, 1]


def fit_s2(tr, te, feats=S2_FEATS):
    """Small LightGBM on the stage-2 features; returns test probabilities."""
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=THREADS, verbose=-1)
    b = lgb.train(params, lgb.Dataset(tr[feats].values.astype(np.float32), tr.label.values), 400)
    return b.predict(te[feats].values.astype(np.float32))


def report(va, prob, tag, work=V2):
    """Tune thresholds for given validation probabilities and print standard and test-like macro F0.5 per country."""
    ents = val_entities(work)
    gt = load_gt(os.path.join(work, "train_gt.parquet"))
    sc = va[["qid", "s1", "label", "country"]].copy()
    sc["prob"] = prob
    std = _tune_all(sc, gt, ents, f"{tag} std")
    tl = _tune_all(testlike(sc, work), gt, ents, f"{tag} test-like")
    print(f"== {tag}: std {std['global']['f05']:.4f} (US {std['US']['f05']:.4f}, India {std['India']['f05']:.4f}) | "
          f"test-like {tl['global']['f05']:.4f} (US {tl['US']['f05']:.4f}, India {tl['India']['f05']:.4f})", flush=True)
    return std, tl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names")
    ap.add_argument("--p1", default="v2")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    va = load_val(a.names.split("+"), a.p1)
    print(f"val pairs {len(va):,}, states {va.st.value_counts().to_dict()}", flush=True)
    skip = set(a.skip.split(","))
    if "v2" not in skip:
        report(va, va.p1.values, f"stage-1 ({a.p1}) alone")
    if "ce" not in skip:
        report(va, 1 / (1 + np.exp(-va.ce.values)), "cross-encoder alone")
    if "lr" not in skip:
        report(va, cv_predict(va, fit_lr), "blend-lr (LOSO)")
    va = stage2_features(va)
    report(va, cv_predict(va, fit_s2), "stage2 LightGBM (LOSO)")
