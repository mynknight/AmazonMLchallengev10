"""France veto: which of v6a's France matches would a cross-encoder drop?

    python veto_fr.py [v ...]        # veto thresholds (CE probability), default 0.05 0.1 0.2

v6a France decisions (fin_nv_u, threshold 0.82) are kept, except matches whose France-adapted CE
probability (mean of the available ce_e5fr_f* folds) is below v.  The plain e5k4 CE is shown for
comparison: it misreads French noise words, so it must not be used as the veto.  Pattern shares use
the query's v6a match: 'noise word' = one extra word from the French noise set, 'vocab swap' = one
common word swapped for another.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from common10 import CE4, WORK10
from ustats import core_of, toks

NOISE = {"groupe", "developpement", "france", "fils", "associes", "services"}


def france_scores():
    pr = pd.read_parquet(os.path.join(CE4, "pairs_test.parquet"), columns=["qid", "s1", "country", "p1"])
    sel = ((pr.country == "France") & (pr.p1 >= 0.01)).values
    fr_files = sorted(glob.glob(os.path.join(WORK10, "ce_fr", "ce_e5fr_f*_test.npy")))
    pr["ce_fr"] = np.nanmean([np.load(f) for f in fr_files], axis=0)
    pr["ce_k4"] = np.nanmean([np.load(os.path.join(CE4, f"ce_e5k4_f{k}_test.npy")) for k in range(4)], axis=0)
    fr = pr[sel]
    v6a = pd.read_parquet(os.path.join(WORK10, "test_fr_v6a.parquet"), columns=["qid", "s1", "prob"])
    d = v6a.merge(fr[["qid", "s1", "ce_fr", "ce_k4"]], on=["qid", "s1"], how="left", validate="one_to_one")
    d["p_fr"] = 1 / (1 + np.exp(-d.ce_fr.values)); d["p_k4"] = 1 / (1 + np.exp(-d.ce_k4.values))
    return d, len(fr_files)


def main(vs):
    d, nf = france_scores()
    b = d.sort_values(["qid", "prob"], ascending=[True, False]).drop_duplicates("qid")
    m = b[b.prob >= 0.82].copy()
    qc, dc = core_of("test", m.qid.values), core_of("test", m.s1.values)
    eq = [toks(a) - toks(c) for a, c in zip(qc, dc)]; ed = [toks(c) - toks(a) for a, c in zip(qc, dc)]
    m["noise"] = [len(a) == 1 and next(iter(a)) in NOISE and len(e) <= 1 for a, e in zip(eq, ed)]
    m["swap"] = [len(a) == 1 and len(e) == 1 and next(iter(a)) not in NOISE for a, e in zip(eq, ed)]
    m["exact"] = [len(a) == 0 and len(e) == 0 for a, e in zip(eq, ed)]
    print(f"v6a France matches: {len(m):,} (France CE folds available: {nf})")
    for col, name in (("p_fr", "France-adapted CE"), ("p_k4", "plain CE (not usable)")):
        for v in vs:
            drop = m[col].values < v
            print(f"  {name:22} veto < {v:<4}: drops {drop.sum():7,} ({drop.mean():.2%}) | share dropped among "
                  f"noise-word {drop[m.noise.values].mean():.3f}, word-swap {drop[m.swap.values].mean():.3f}, "
                  f"exact-name {drop[m.exact.values].mean():.4f}")
    m.to_parquet(os.path.join(WORK10, "fr_veto_candidates.parquet"), index=False)


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [0.05, 0.1, 0.2])
