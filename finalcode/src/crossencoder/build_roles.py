"""Compute the label-free word-role features (wordroles.py) for the train, val and test pair lists.

    python build_roles.py

Writes work/u_{train,val,test}.parquet, row-aligned with work/pairs_{train,val,test}.parquet.
Train / validation rows use the statistics of the training split (US, India); test rows use the
statistics of the test split, per country (France included).
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import V2, WORK
from wordroles import Tokens, role_features, role_tables, show

CLEAN_COLS = ["a_tset", "num_first", "a_empty_q"]
WORDS = ["center", "services", "service", "partners", "enterprises", "ventures", "group", "holdings",
         "technology", "trading", "industries", "solutions", "fils", "groupe", "associes", "developpement",
         "france", "club", "ecole", "amicale", "comite", "sportive", "union", "pharmacie", "parents"]


def feat_rows(split, rows):
    """Selected rows (by position) of the v2 feature file of a split, only the columns needed to find clean pairs."""
    f = pq.read_table(os.path.join(V2, f"feat_{split}.parquet"), columns=CLEAN_COLS).to_pandas()
    return f.iloc[rows].reset_index(drop=True)


def main():
    """Build the label-free word-role tables per country and split (train+val for US/India, test candidates for all countries) and write u_train/u_val/u_test.parquet."""
    t0 = time.time()
    # ---------------- training split (US, India): statistics from the training + validation pairs
    tok = Tokens("train")
    tr = pd.read_parquet(os.path.join(WORK, "pairs_train.parquet"), columns=["qid", "s1", "p1", "country", "row"])
    va = pd.read_parquet(os.path.join(WORK, "pairs_val.parquet"), columns=["qid", "s1", "p1", "country", "row"])
    tr = pd.concat([tr, feat_rows("train", tr.row.values)], axis=1)
    va = pd.concat([va, feat_rows("val", va.row.values)], axis=1)
    tables = role_tables(tok, pd.concat([tr, va], ignore_index=True))
    print({c: (t["n_clean"], t["n_s1"]) for c, t in tables.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    print(show(tok, tables, WORDS).to_string(index=False), flush=True)
    for name, df in (("train", tr), ("val", va)):
        u = role_features(tok, tables, df.qid.values, df.s1.values, df.country.values)
        u.to_parquet(os.path.join(WORK, f"u_{name}.parquet"), index=False)
        print(f"u_{name}: {len(u):,} rows, NaN share {u.isna().mean().round(3).to_dict()} ({time.time() - t0:.0f}s)",
              flush=True)
    del tok, tr, va
    # ---------------- test split (US, India, France): statistics from the test candidates
    tok = Tokens("test")
    te = pd.read_parquet(os.path.join(WORK, "pairs_test.parquet"), columns=["qid", "s1", "p1", "country"])
    ft = pd.read_parquet(os.path.join(WORK, "feat_test_p1.parquet"), columns=["qid", "s1"] + CLEAN_COLS)
    te = te.merge(ft, on=["qid", "s1"], how="left", validate="one_to_one")
    assert not te[CLEAN_COLS].isna().any().any()
    tables = role_tables(tok, te)
    print({c: (t["n_clean"], t["n_s1"]) for c, t in tables.items()}, f"({time.time() - t0:.0f}s)", flush=True)
    print(show(tok, tables, WORDS).to_string(index=False), flush=True)
    u = role_features(tok, tables, te.qid.values, te.s1.values, te.country.values)
    u.to_parquet(os.path.join(WORK, "u_test.parquet"), index=False)
    print(f"u_test: {len(u):,} rows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
