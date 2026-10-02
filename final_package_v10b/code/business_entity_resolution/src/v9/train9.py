"""Final-stage LightGBM variants with the language-agnostic u_* word features (ustats.py).

    python train9.py <name> [--ce e5k4_f0+e5k4_f1+e5k4_f2+e5k4_f3] [--no_w] [--u] [--ea]

Same rows, parameters, CE handling and test-like evaluation as crossencoder/src/train_final.py, so
numbers are directly comparable with fin_tr (0.9900) / fin_noce (0.9859) / fin_nv_noce.
Writes work/<name>.txt, work/val_scores_<name>.parquet, work/<name>_thresholds.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common9 import CE_WORK, THREADS, WORK9
import train_final as tf
from evaluate import report
from pairs import feature_cols
from ustats import FEATS as UFEATS



def empty_address_queries(split="train"):
    """S2/S3 records of US/India with an empty address: their candidates come from ea.py."""
    from common9 import V2
    r = pd.concat([pd.read_parquet(os.path.join(V2, f"{split}_s{k}_norm.parquet"), columns=["entity_id", "country", "state", "ad", "nums"])
                   for k in (2, 3)], ignore_index=True)
    s1 = pd.read_parquet(os.path.join(V2, f"{split}_s1_norm.parquet"), columns=["country", "state"])
    s1_states = set(zip(s1.country, s1.state)) - {(c, "") for c in s1.country.unique()}
    # only queries that ea.py re-blocked: empty address AND searched country-wide (no S1 state)
    fallback = ~pd.Series(list(zip(r.country, r.state))).isin(s1_states).values
    return set(r.entity_id[(r.ad == "") & (r.nums == "") & r.country.isin(["US", "India"]) & fallback])


_EMPTY = None


def with_ea(df, split, p1_min, use_u):
    """Replace the rows of empty-address queries by the deeper-retrieval rows of ea.py; CE context
    features are recomputed over the combined set."""
    global _EMPTY
    _EMPTY = _EMPTY or empty_address_queries()
    ea_dir = os.path.join(WORK9, "ea")
    new = pd.read_parquet(os.path.join(ea_dir, f"rows_{split}.parquet"))
    new["ce"] = np.load(os.path.join(ea_dir, f"ce_{split}.npy"))
    if use_u:
        un = pd.read_parquet(os.path.join(ea_dir, f"u_{split}.parquet"))
        for c in UFEATS:
            new[c] = un[c].values
    new = new[new.p1.values >= p1_min]
    old = df.qid.isin(_EMPTY).values
    n_true_old, n_true_new = int(df.label.values[old].sum()), int(new.label.sum())
    df = pd.concat([df[~old], new[[c for c in df.columns if c in new.columns]]], ignore_index=True)
    df = tf.add_ce(df, df.ce.values)
    print(f"  ea {split}: replaced {old.sum():,} rows ({n_true_old:,} true) by {len(new):,} ({n_true_new:,} true)", flush=True)
    return df


def load_matrix(ce, u=True, ea=True, no_w=True):
    """Training / validation frames and the feature list of the final model (as k4_u_ea)."""
    tr = tf.load_train(ce)
    va = tf.load_val(ce)
    if u:
        ut = pd.read_parquet(os.path.join(WORK9, "u_train.parquet"))
        uv = pd.read_parquet(os.path.join(WORK9, "u_val.parquet")).iloc[va["_pos"].values].reset_index(drop=True)
        for c in UFEATS:
            tr[c] = ut[c].values
            va[c] = uv[c].values
        del ut, uv
    va_ids = pd.read_parquet(os.path.join(CE_WORK, "pairs_val.parquet"), columns=IDS).iloc[va["_pos"].values]
    for c in IDS:
        va[c] = va_ids[c].values
    if ea:
        tr = with_ea(tr, "train", 1e-4, u)
        va = with_ea(va, "val", tf.CAND_P1, u)
    drop = set(tf.CE_SIZE) | {"fold", "p1"}
    if no_w:
        drop |= set(tf.WORDS)
    feats = [f for f in feature_cols(tr) if f not in drop]
    if u:
        feats += [f for f in UFEATS if f not in feats]
    return tr, va, feats


IDS = ["qid", "s1", "label", "country"]


def pseudo_test_rows(ce, cols, pos=0.98, neg=0.02, scores="test_usin_k4_u_ea.parquet"):
    """Confident US/India test decisions of the production model as extra training rows (self-training):
    best candidate of its query with prob >= pos -> 1, prob <= neg -> 0; the uncertain band is left out.
    Features are built exactly as at scoring time (predict9.load_test, deeper retrieval included)."""
    from predict9 import load_test
    m = load_test(ce, os.path.join(WORK9, "ce4"), ["US", "India"], ea=True)
    sc = pd.read_parquet(os.path.join(WORK9, scores), columns=["qid", "s1", "prob"])
    m = m.merge(sc, on=["qid", "s1"], how="inner", validate="one_to_one")
    top = ~m.sort_values(["qid", "prob"], ascending=[True, False]).qid.duplicated().reindex(m.index).values
    m["label"] = np.where(top & (m.prob.values >= pos), 1, np.where(m.prob.values <= neg, 0, -1)).astype(np.int8)
    m = m[m.label >= 0]
    print(f"  self-training: {int((m.label == 1).sum()):,} pseudo-positive, {int((m.label == 0).sum()):,} pseudo-negative test rows", flush=True)
    return m[[c for c in cols if c in m.columns]]


def main(a):
    import lightgbm as lgb
    ce = [n for n in a.ce.split("+") if n]
    t0 = time.time()
    tr, va, feats = load_matrix(ce, a.u, a.ea, a.no_w)
    if a.self_train:
        tr = pd.concat([tr, pseudo_test_rows(ce, list(tr.columns))], ignore_index=True)
    ids_cols = IDS
    print(f"[{a.name}] train {len(tr):,} val {len(va):,} feats {len(feats)} ce {ce} ({time.time() - t0:.0f}s)", flush=True)
    params = dict(tf.PARAMS, num_threads=THREADS, seed=a.seed)
    dtr = lgb.Dataset(tr[feats].values.astype(np.float32), tr.label.values, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(va[feats].values.astype(np.float32), va.label.values, reference=dtr)
    del tr
    b = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    b.save_model(os.path.join(WORK9, f"{a.name}.txt"))
    imp = pd.Series(b.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).head(20).round(4).to_string(), flush=True)
    from decide import apply_rules
    prob = apply_rules(va, b.predict(va[feats].values.astype(np.float32), num_threads=THREADS))
    ids = va[ids_cols].reset_index(drop=True)
    ids["prob"] = prob.astype(np.float32)
    ids.to_parquet(os.path.join(WORK9, f"val_scores_{a.name}.parquet"), index=False)
    std, tl = report(ids.assign(st=None), prob, a.name)
    res = dict(tl); res["tuned_on"] = "testlike"; res["std"], res["testlike"] = std, tl
    json.dump(res, open(os.path.join(WORK9, f"{a.name}_thresholds.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--ce", default="")
    ap.add_argument("--no_w", action="store_true")
    ap.add_argument("--u", action="store_true")
    ap.add_argument("--self_train", action="store_true", help="add confident US/India test pseudo-labels")
    ap.add_argument("--ea", action="store_true", help="deeper retrieval for empty-address queries (ea.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4000)
    main(ap.parse_args())
