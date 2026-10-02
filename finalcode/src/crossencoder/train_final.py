"""Final matcher: LightGBM on the v2 hand features + cross-encoder features, over the stage-1
filtered candidates (the cascade: TF-IDF blocking -> v2 LightGBM filter -> this model).

    python train_final.py <name> <ce_names> [--drop w_*] [--no-ce]

<ce_names>: '+'-joined cross-encoder runs, one per fold (e.g. e5s_f0+e5s_f1).  Training rows get
the out-of-fold score (the model trained on the other fold of source-1 states); validation and
test rows get the mean over the fold models.
CE features: ce (logit), its rank / gap to the best other candidate within the query and within
the source-1 entity.  The stage-1 probability itself is NOT a feature: it is in-sample on the
training rows.
Writes work/<name>.txt, work/val_scores_<name>.parquet, work/<name>_thresholds.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common import THREADS, V2, WORK
from evaluate import context, report
from pairs import feature_cols

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=THREADS, verbose=-1, seed=0)
WORDS = ["w_q_max", "w_q_min", "w_q_n", "w_d_max", "w_d_min", "w_d_n"]
CE_FEATS = ["ce", "q_ce_rk", "q_ce_gap", "d_ce_rk", "d_ce_gap"]
# sizes of the filtered candidate lists: their distribution differs between the training rows
# (in-sample stage-1 filter) and validation / test (honest filter), so they are not used
CE_SIZE = ["q_ce_n", "d_ce_n"]
CAND_P1 = float(os.environ.get("CE_CAND_P1", 0.01))  # final candidate set: stage-1 p1 >= CAND_P1


def ce_scores(split, names, fold=None):
    """Cross-encoder scores of a split: train = the out-of-fold score of each row, val/test = mean over the fold models."""
    arr = [np.load(os.path.join(WORK, f"ce_{n}_{split}.npy")) for n in names]
    if split == "train":  # out-of-fold: each row has exactly one non-NaN score among the fold models
        out = np.full(len(arr[0]), np.nan, np.float32)
        for a in arr:
            ok = ~np.isnan(a)
            out[ok] = a[ok]
        return out
    return np.mean(arr, axis=0).astype(np.float32)


def add_ce(df, ce):
    """Add the CE score and its context features (rank and gap within the query and within the S1 entity)."""
    df["ce"] = ce
    df = context(df, "ce", "qid", "q_ce")
    df = context(df, "ce", "s1", "d_ce")
    return df


def rows_of(split, rows):
    """v2 features (+ extra-word features) of feat_<split>.parquet rows."""
    f = pd.read_parquet(os.path.join(V2, f"feat_{split}.parquet"))
    a = pd.read_parquet(os.path.join(V2, f"feat_{split}_aug_tok.parquet"))
    for c in a.columns:
        f[c] = a[c].values
    del a
    return f.iloc[rows].reset_index(drop=True)


def add_u(df, split, keep=None):
    """label-free word-role features (build_roles.py), row-aligned with pairs_<split>.parquet"""
    u = pd.read_parquet(os.path.join(WORK, f"u_{split}.parquet"))
    if keep is not None:
        u = u[keep]
    for c in u.columns:
        df[c] = u[c].values
    return df


def load_train(ce_names, use_u=False):
    """Load the filtered training rows with v2 features, fold ids, optional word-role features and out-of-fold CE scores."""
    pr = pd.read_parquet(os.path.join(WORK, "pairs_train.parquet"), columns=["row", "fold", "p1"])
    df = rows_of("train", pr.row.values)
    df["fold"] = pr.fold.values
    if use_u:
        df = add_u(df, "train")
    if ce_names:
        df = add_ce(df, ce_scores("train", ce_names))
        assert not df.ce.isna().any(), "missing out-of-fold cross-encoder scores"
    return df


def load_val(ce_names, cand_p1=CAND_P1, use_u=False):
    """Validation rows of the final candidate set (p1 >= cand_p1); CE context within that set."""
    pr = pd.read_parquet(os.path.join(WORK, "pairs_val.parquet"), columns=["row", "p1"])
    keep = pr.p1.values >= cand_p1
    df = rows_of("val", pr.row.values[keep])
    df["p1"] = pr.p1.values[keep]
    df["_pos"] = np.nonzero(keep)[0]
    if use_u:
        df = add_u(df, "val", keep)
    if ce_names:
        df = add_ce(df, ce_scores("val", ce_names)[keep])
    return df


def main(a):
    """Train the final LightGBM (v2 features + optional CE and word-role features, optional dropped features), early-stop on validation, save the model and tune thresholds."""
    import lightgbm as lgb
    ce_names = [] if a.no_ce else a.ce.split("+")
    t0 = time.time()
    tr = load_train(ce_names, a.u)
    va = load_val([n for n in a.val_ce.split("+") if n] or ce_names if ce_names else [], use_u=a.u)
    drop = set(x for x in a.drop.split(",") if x)
    if "w_*" in drop:
        drop |= set(WORDS)
    feats = [f for f in feature_cols(tr) if f not in drop and f not in ["fold", "p1"] + CE_SIZE]
    print(f"[{a.name}] train rows {len(tr):,} (pos {tr.label.mean():.3f}), val rows {len(va):,}, "
          f"{len(feats)} features, CE {ce_names} ({time.time() - t0:.0f}s)", flush=True)
    dtr = lgb.Dataset(tr[feats].values.astype(np.float32), tr.label.values, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(va[feats].values.astype(np.float32), va.label.values, reference=dtr)
    del tr
    b = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    b.save_model(os.path.join(WORK, f"{a.name}.txt"))
    imp = pd.Series(b.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).head(15).round(4).to_string(), flush=True)
    from decide import apply_rules
    prob = apply_rules(va, b.predict(va[feats].values.astype(np.float32), num_threads=THREADS))
    va_ids = pd.read_parquet(os.path.join(WORK, "pairs_val.parquet"), columns=["qid", "s1", "label", "country"])
    va_ids = va_ids.iloc[va["_pos"].values].reset_index(drop=True)
    va_ids["prob"] = prob.astype(np.float32)
    va_ids.to_parquet(os.path.join(WORK, f"val_scores_{a.name}.parquet"), index=False)
    std, tl = report(va_ids.assign(st=None), prob, a.name)
    res = dict(tl)
    res["tuned_on"] = "testlike"
    res["std"], res["testlike"] = std, tl
    with open(os.path.join(WORK, f"{a.name}_thresholds.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("ce", nargs="?", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--no_ce", action="store_true")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--val_ce", default="", help="CE runs averaged for validation (default: all)")
    ap.add_argument("--u", action="store_true", help="add the label-free word-role features (wordroles.py)")
    main(ap.parse_args())
