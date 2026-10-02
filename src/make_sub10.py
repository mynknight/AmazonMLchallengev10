"""Write a V10 submission: v9c (V9 US/India + v6a France) with the France cross-encoder veto.

    python make_sub10.py <out_dir> [--v 0.2] [--classes swap] [--add 0.6:0.98,0.3:0.99]

A v6a France match (best candidate of its query, prob >= 0.82) is dropped (the query is left
unmatched) when the France-adapted CE probability (mean of ce_e5fr_f0..3) is below --v AND the name
difference is in --classes.  Classes compare the core-name token sets after runs of single letters are
joined ("s c i" -> "sci"):
    exact   same tokens                     (CE disagrees on the address: not vetoed by default)
    subset  one side adds words             (short or noise-word names: not vetoed by default)
    typo    one word each, edit-similar     (not vetoed)
    swap    one word each, unrelated        (lycee <-> club: the pattern v6a is built to reject)
    other   several words differ
Additions (--add lo:ce tiers): a query v6a leaves unmatched is matched to its best candidate when that
candidate's v6a prob is in [lo, 0.82), its France CE prob >= the ce of the highest tier with lo <= prob,
and its class is in --add_classes.  On US/India validation, model-best matches confirmed by CE >= 0.99
are 94-99% true even at model prob 0.1-0.6 (v10b = --add 0.6:0.98 gained +0.00028 on the leaderboard).
US/India rows are identical to ~/MLchallenge_v9/output/v9c_v6afr.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from common10 import V10, WORK10
from ustats import core_of, toks
from veto_fr import france_scores

V9 = "/home/mayankc/MLchallenge_v9"
FR_THR = 0.82


def join_letters(core):
    out, run = [], ""
    for w in (core.split() if isinstance(core, str) else []):
        if len(w) == 1:
            run += w
            continue
        if run:
            out.append(run); run = ""
        out.append(w)
    if run:
        out.append(run)
    return " ".join(out)


def name_class(a, c):
    A, C = toks(join_letters(a)), toks(join_letters(c))
    ea, ec = A - C, C - A
    if not ea and not ec:
        return "exact"
    if len(ea) == 1 and len(ec) == 1:
        x, y = next(iter(ea)), next(iter(ec))
        return "typo" if Levenshtein.distance(x, y) <= 1 or fuzz.ratio(x, y) >= 80 else "swap"
    if not ea or not ec:
        return "subset"
    return "other"


def vetoed(v, classes, tiers, add_classes):
    d, nf = france_scores()
    assert nf == 4, f"expected 4 France CE folds, found {nf}"
    b = d.sort_values(["qid", "prob"], ascending=[True, False]).drop_duplicates("qid")
    add_lo = min([lo for lo, _ in tiers], default=FR_THR)
    m = b[b.prob >= min(FR_THR, add_lo)].copy()
    m["cls"] = [name_class(a, c) for a, c in zip(core_of("test", m.qid.values), core_of("test", m.s1.values))]
    need = np.full(len(m), np.inf)
    for lo, ce in sorted(tiers):                      # ascending lo: the highest tier <= prob wins
        need[m.prob.values >= lo] = ce
    m["add"] = (m.prob.values < FR_THR) & (m.p_fr.values >= need) & m.cls.isin(add_classes).values
    for lo, ce in sorted(tiers, reverse=True):
        t = m["add"].values & (need == ce) & (m.prob.values >= lo)
        print(f"add tier prob >= {lo}, CE >= {ce}: {t.sum():,} pairs", flush=True)
    if tiers:
        print(f"add total {m['add'].sum():,} ({add_classes}); by class {m[m['add']].cls.value_counts().to_dict()}", flush=True)
    m = m[(m.prob.values >= FR_THR) | m["add"].values].copy()
    m["veto"] = (m.prob.values >= FR_THR) & (m.p_fr.values < v) & m.cls.isin(classes).values
    print(f"v6a France matches {len(m):,}; class shares {m.cls.value_counts(normalize=True).round(4).to_dict()}")
    print(f"veto CE < {v} on {classes}: drops {m.veto.sum():,} ({m.veto.mean():.3%}); by class "
          f"{m[m.veto].cls.value_counts().to_dict()}", flush=True)
    return d, m


def main(a):
    classes = a.classes.split(",")
    tiers = [tuple(map(float, t.split(":"))) for t in a.add.split(",") if t]
    d, m = vetoed(a.v, classes, tiers, a.add_classes.split(","))
    tag = os.path.basename(os.path.normpath(a.out))
    m.to_parquet(os.path.join(WORK10, f"veto_{tag}.parquet"), index=False)
    fr = pd.read_parquet(os.path.join(WORK10, "test_fr_v6a.parquet"))
    fr.loc[fr.qid.isin(m.qid[m.veto]).values, "prob"] = 0.0      # whole query unmatched, no fallback
    add = pd.MultiIndex.from_frame(m.loc[m["add"], ["qid", "s1"]])
    fr.loc[pd.MultiIndex.from_frame(fr[["qid", "s1"]]).isin(add), "prob"] = FR_THR   # best candidate -> accepted
    fr_path = os.path.join(WORK10, f"test_fr_{tag}.parquet")
    fr.to_parquet(fr_path, index=False)
    out = os.path.abspath(a.out)
    subprocess.run([sys.executable, "make_sub9.py", out,
                    "--usin", f"{V9}/work/test_usin_k4_u_ea.parquet", "--usin_thr", f"{V9}/work/k4_u_ea_thresholds.json",
                    "--fr", "new", "--fr_new", fr_path, "--fr_thr", str(FR_THR)], cwd=f"{V9}/src", check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v", type=float, default=0.2)
    ap.add_argument("--classes", default="swap")
    ap.add_argument("--add", default="", help="lo:ce tiers, e.g. 0.6:0.98,0.3:0.99 (empty = no additions)")
    ap.add_argument("--add_classes", default="exact,typo,subset,other")
    main(ap.parse_args())
