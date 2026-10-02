"""Score the test candidate set (stage-1 p1 >= 0.01, the V5a/V7 final set) with a V9 model.

    python predict9.py <model_name> <out.parquet> [--ce e5tr_f0+e5tr_f1] [--ce_dir DIR] [--countries France]

Rows = crossencoder/work/feat_test_p1.parquet (+ u_test.parquet, row-aligned) joined with CE test
scores (pairs_test.parquet order).  Only rows of --countries are written.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common9 import CE_WORK, THREADS, WORK9
from decide import apply_rules
from train_final import CAND_P1, add_ce
from ustats import FEATS as UFEATS


def load_test(ce, ce_dir, countries, ea=False):
    feat = pd.read_parquet(os.path.join(CE_WORK, "feat_test_p1.parquet"))
    u = pd.read_parquet(os.path.join(WORK9, "u_test.parquet"))
    for c in UFEATS:
        feat[c] = u[c].values
    del u
    if ce:
        # CE scores follow pairs_test.parquet order, which differs from feat_test_p1: join on the pair
        pr = pd.read_parquet(os.path.join(CE_WORK, "pairs_test.parquet"), columns=["qid", "s1"])
        pr["ce_raw"] = np.nanmean([np.load(os.path.join(ce_dir, f"ce_{n}_test.npy")) for n in ce], axis=0)
        n0 = len(feat)
        feat = feat.merge(pr, on=["qid", "s1"], how="inner", validate="one_to_one")
        assert len(feat) == n0 == len(pr), (n0, len(feat), len(pr))
    keep = feat.p1.values >= CAND_P1
    if countries:
        keep &= feat.country.isin(countries).values
    m = feat[keep].reset_index(drop=True)
    if ea:  # deeper retrieval for empty-address queries: replace their rows by ea.py's (US / India)
        from train9 import empty_address_queries
        ea_dir = os.path.join(WORK9, "ea")
        new = pd.read_parquet(os.path.join(ea_dir, "rows_test.parquet"))
        un = pd.read_parquet(os.path.join(ea_dir, "u_test.parquet"))
        for c in UFEATS:
            new[c] = un[c].values
        new["ce_raw"] = np.load(os.path.join(ea_dir, "ce_test.npy"))
        new = new[(new.p1.values >= CAND_P1) & new.country.isin(countries or new.country.unique()).values]
        old = m.qid.isin(empty_address_queries("test")).values
        print(f"ea: replaced {old.sum():,} test rows by {len(new):,}", flush=True)
        m = pd.concat([m[~old], new[[c for c in m.columns if c in new.columns]]], ignore_index=True)
    if ce:
        m = add_ce(m, m.pop("ce_raw").values)
    return m


def main(a):
    import lightgbm as lgb
    ce = [n for n in a.ce.split("+") if n]
    m = load_test(ce, a.ce_dir, [c for c in a.countries.split(",") if c], a.ea)
    b = lgb.Booster(model_file=os.path.join(WORK9, f"{a.model}.txt"))
    prob = apply_rules(m, b.predict(m[b.feature_name()].values.astype(np.float32), num_threads=THREADS))
    out = m[["qid", "s1", "country", "p1"]].copy()
    out["prob"] = prob.astype(np.float32)
    out.to_parquet(a.out, index=False)
    print(f"{a.model}: {len(out):,} pairs scored -> {a.out}", out.groupby("country").size().to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--ce", default="")
    ap.add_argument("--ce_dir", default=CE_WORK)
    ap.add_argument("--countries", default="")
    ap.add_argument("--ea", action="store_true")
    main(ap.parse_args())
