"""Train the LightGBM matcher and tune the decision threshold for macro F0.5.

    python train.py <work> [model_name]
    python train.py <work> <model_name> eval      # re-tune from saved validation scores

Env knobs for experiments (defaults reproduce the plain run):
  ER_TRAIN_SUB  keep only training queries with hash < value (subset of the train_rate sample)
  ER_AUG        feature groups from augment.py to add / replace, e.g. "ph2,x"
  ER_DROP       feature columns to leave out, e.g. "ph_tsort,ph_tset"
  ER_TRAIN_DW   sample weight of rows that belong to distractor queries (default 1)
  ER_DISTRACTOR_W  weight of validation distractor queries in the "test-like" evaluation
                (default 1.9: the test set has ~40% distractor records vs 26% in train)
  ER_TUNE_ON    "testlike" (default) or "std": which evaluation the saved decision rule is tuned on
  ER_ROUNDS     maximum boosting rounds (default 2000; early stopping on validation log-loss)
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from decide import apply_rules, tune_ef, tune_fine
from metrics import load_gt, macro_f05
from pairs import feature_cols
from pipeline import VAL_STATES, _gt_inverse, _hash01

THREADS = int(os.environ.get("ER_THREADS", os.cpu_count()))
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=THREADS, verbose=-1)


def val_entities(work):
    s1 = pd.read_parquet(os.path.join(work, "train_s1_norm.parquet"), columns=["entity_id", "country", "state"])
    m = np.zeros(len(s1), bool)
    for c, sts in VAL_STATES.items():
        m |= (s1.country.values == c) & s1.state.isin(sts).values
    return s1.loc[m, ["entity_id", "country"]]


def _env_list(name):
    return [x for x in os.environ.get(name, "").split(",") if x]


def load_split(work, split):
    """feat_<split>.parquet plus the augment.py groups named in ER_AUG (columns added / replaced)."""
    df = pd.read_parquet(os.path.join(work, f"feat_{split}.parquet"))
    for g in _env_list("ER_AUG"):
        aug = pd.read_parquet(os.path.join(work, f"feat_{split}_aug_{g}.parquet"))
        assert len(aug) == len(df), f"augment group {g} is not aligned with feat_{split}"
        for c in aug.columns:
            df[c] = aug[c].values
    return df


def main(work, name="lgb_v1", rounds=2000):
    tr = load_split(work, "train")
    sub = float(os.environ.get("ER_TRAIN_SUB", 0))
    if sub:
        # keep whole queries whose hash < sub: exactly the rows `pipeline.py features` writes
        # with train_rate=sub, so fast experiments need no feature rebuild
        uq = tr.qid.unique()
        tr = tr[tr.qid.isin(uq[_hash01(uq) < sub])].reset_index(drop=True)
    va = load_split(work, "val")
    drop = set(_env_list("ER_DROP"))
    feats = [f for f in feature_cols(tr) if f not in drop]
    print(f"train rows {len(tr):,} (pos {tr.label.mean():.3f}), val rows {len(va):,}; {len(feats)} features")
    weight = None
    dw = float(os.environ.get("ER_TRAIN_DW", 1))
    if dw != 1:
        # up-weight every row of a distractor query (a query with no true S1): test has more of them
        weight = np.where(tr.qid.isin(_gt_inverse(work).index).values, 1.0, dw).astype(np.float32)
    dtr = lgb.Dataset(tr[feats].values.astype(np.float32), tr.label.values, weight=weight,
                      feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(va[feats].values.astype(np.float32), va.label.values, reference=dtr)
    del tr
    booster = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    path = os.path.join(work, f"{name}.txt")
    booster.save_model(path)
    imp = pd.Series(booster.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print(imp.head(25).round(0).to_string())

    va["prob"] = apply_rules(va, booster.predict(va[feats].values.astype(np.float32), num_threads=THREADS))
    sc = va[["qid", "s1", "prob", "label", "country"]]
    sc.to_parquet(os.path.join(work, f"val_scores_{name}.parquet"))
    evaluate(work, sc, name)


def testlike(sc, work, w=None):
    """Validation scores re-weighted to the test distractor share: each distractor query (a query
    with no true S1) gets copies with identical candidates and probabilities -- floor(w-1) of them,
    plus one more for a hash-chosen fraction frac(w-1) -- so FPs from distractors count ~w times."""
    w = float(os.environ.get("ER_DISTRACTOR_W", 1.9)) if w is None else w
    qids = sc.qid.unique()
    dis = qids[~pd.Index(qids).isin(_gt_inverse(work).index)]
    h = _hash01(dis)
    parts, k = [sc], 0
    while w - 1 - k > 0:
        sel = dis[h < min(1.0, w - 1 - k)]
        cp = sc[sc.qid.isin(sel)].copy()
        cp["qid"] = cp.qid + f"~{k}"
        parts.append(cp)
        k += 1
    return pd.concat(parts, ignore_index=True)


def _tune_all(sc, gt, ents, tag):
    res = {}
    for country, g in list(ents.groupby("country")) + [("global", ents)]:
        ids = g.entity_id.tolist()
        gsub = {s: gt[s] for s in ids}
        part = sc if country == "global" else sc[sc.country == country]
        t, f = tune_fine(part, gsub, ids)
        (mi, fl), fe = tune_ef(part, gsub, ids)
        res[country] = {"threshold": float(t), "f05": f, "ef_miss": mi, "ef_floor": fl, "ef_f05": fe}
        print(f"[{tag}] {country}: entities {len(ids):,} | threshold {t:.2f} -> F0.5 {f:.4f} | "
              f"expected-F (miss {mi}, floor {fl}) -> F0.5 {fe:.4f}", flush=True)
    res["rule"] = "ef" if res["global"]["ef_f05"] > res["global"]["f05"] else "threshold"
    return res


def evaluate(work, sc, name):
    ents = val_entities(work)
    gt = load_gt(os.path.join(work, "train_gt.parquet"))
    std = _tune_all(sc, gt, ents, "std")
    tl = _tune_all(testlike(sc, work), gt, ents, "test-like")
    on = os.environ.get("ER_TUNE_ON", "testlike")
    res = dict(tl if on == "testlike" else std)
    res["tuned_on"] = on
    res["std"], res["testlike"] = std, tl
    g = lambda r: max(r["global"]["f05"], r["global"]["ef_f05"])
    print(f"SUMMARY {name}: F0.5 std {g(std):.4f} (rule {std['rule']}) | test-like {g(tl):.4f} (rule {tl['rule']})"
          f" | submission rule tuned on {on}: {res['rule']}", flush=True)
    with open(os.path.join(work, f"{name}_thresholds.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "eval":
        sc = pd.read_parquet(os.path.join(sys.argv[1], f"val_scores_{sys.argv[2]}.parquet"))
        evaluate(sys.argv[1], sc, sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "lgb_v1", int(os.environ.get("ER_ROUNDS", 2000)))
