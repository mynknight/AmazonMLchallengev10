"""Score the test candidates with the final matcher(s).

    python predict_final.py <out_scores.parquet> <model_seen> [<model_unseen>] --ce e5tr_f0+e5tr_f1

Rows = feat_test_p1.parquet (v2 features of the pairs that pass the stage-1 filter, feat_test.py)
joined with the cross-encoder test scores (pairs_test.parquet order, mean over the fold models).
Countries seen in training use <model_seen>; others (France) use <model_unseen> when given, else
their stage-1 probability is kept (i.e. the v3 decision for that country).
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import THREADS, V2, WORK
from decide import apply_rules
from run_blocking import countries_of
from train_final import CAND_P1, add_ce


def main(a):
    import lightgbm as lgb
    feat = pd.read_parquet(os.path.join(WORK, "feat_test_p1.parquet"))
    pr = pd.read_parquet(os.path.join(WORK, "pairs_test.parquet"), columns=["qid", "s1", "p1"])
    if a.u:  # label-free word-role features, row-aligned with pairs_test (build_roles.py)
        u = pd.read_parquet(os.path.join(WORK, "u_test.parquet"))
        for c in u.columns:
            pr[c] = u[c].values
    names = [n for n in a.ce.split("+") if n]
    if names:
        pr["ce"] = np.mean([np.load(os.path.join(WORK, f"ce_{n}_test.npy")) for n in names], axis=0)
    key = ["qid", "s1"]
    m = feat.merge(pr.rename(columns={"p1": "p1_v3"}), on=key, how="inner", validate="one_to_one")
    assert len(m) == len(feat) == len(pr), (len(m), len(feat), len(pr))
    print(f"test pairs {len(m):,}; max |p1 - p1(v3)| = {np.abs(m.p1 - m.p1_v3).max():.2e}", flush=True)
    m = m[m.p1 >= CAND_P1].reset_index(drop=True)  # the final candidate set
    print(f"final candidate set (p1 >= {CAND_P1:g}): {len(m):,} pairs", flush=True)
    if names:
        ce = m.pop("ce").values
        m = add_ce(m, ce)
    seen = set(countries_of(V2, "train"))
    prob = m.p1.values.astype(np.float64).copy()
    for path, is_seen in ((a.model_seen, True), (a.model_unseen, False)):
        if not path:
            continue
        b = lgb.Booster(model_file=path)
        sel = m.country.isin(seen).values if is_seen else ~m.country.isin(seen).values
        if sel.any():
            prob[sel] = apply_rules(m[sel], b.predict(m.loc[sel, b.feature_name()].values.astype(np.float32),
                                                      num_threads=THREADS))
            print(f"  {os.path.basename(path)}: {sel.sum():,} pairs", flush=True)
    out = m[["qid", "s1", "country", "p1"]].copy()
    out["prob"] = prob.astype(np.float32)
    out.to_parquet(a.out, index=False)
    print("written", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("model_seen")
    ap.add_argument("model_unseen", nargs="?")
    ap.add_argument("--ce", default="")
    ap.add_argument("--u", action="store_true")
    main(ap.parse_args())
