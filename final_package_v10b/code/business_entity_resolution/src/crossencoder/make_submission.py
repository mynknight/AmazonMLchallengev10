"""Write matching_results.tsv and candidate_pairs.tsv for the cascade.

    python make_submission.py <scores.parquet> <out_dir> <thresholds.json> [<unseen_thresholds.json>]
                              [--cand_p1 1e-4]

<scores.parquet>: one row per test pair scored by the final model (qid, s1, country, p1, prob).
candidate_pairs.tsv = the pairs the final model ran inference on (stage-1 probability >= cand_p1);
matching_results.tsv = best source-1 entity per query, kept if prob >= the country threshold
(countries unseen in training use the 'global' entry of <unseen_thresholds.json>).
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from common import DATA, HOME
from io_utils import read_tsv
from run_blocking import countries_of
from common import V2


def pick(te, thr, unseen_thr, seen):
    t = np.empty(len(te))
    for c in te.country.unique():
        src = unseen_thr if (unseen_thr is not None and c not in seen) else thr
        t[(te.country == c).values] = src.get(c, src["global"])["threshold"]
    te = te.assign(t=t).sort_values(["qid", "prob"], ascending=[True, False])
    best = te.drop_duplicates("qid", keep="first")
    return best[best.prob >= best.t]


def write_lists(path, header, s1_ids, s1, ids):
    grp = pd.Series(ids).groupby(s1).agg(lambda x: ",".join(dict.fromkeys(x)))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{grp.get(sid, '')}\n")


def main(a):
    te = pd.read_parquet(a.scores, columns=["qid", "s1", "country", "p1", "prob"])
    te = te[te.p1 >= a.cand_p1].reset_index(drop=True)
    thr = json.load(open(a.thr))
    unseen = json.load(open(a.unseen_thr)) if a.unseen_thr else None
    seen = set(countries_of(V2, "train"))
    m = pick(te, thr, unseen, seen)
    s1_ids = read_tsv(os.path.join(DATA, "test", "test_source1.tsv"), usecols=["entity_id"]).entity_id.values
    os.makedirs(a.out, exist_ok=True)
    cand = te.sort_values(["s1", "qid"])
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), "source1_entity_id\tcandidate_entity_ids",
                s1_ids, cand.s1.values, cand.qid.values)
    m = m.sort_values(["s1", "prob"], ascending=[True, False])
    write_lists(os.path.join(a.out, "matching_results.tsv"), "source1_entity_id\tmatched_entity_ids",
                s1_ids, m.s1.values, m.qid.values)
    n1 = len(s1_ids)
    print(f"candidates: {len(te):,} pairs = {len(te) / n1:.2f} per S1 entity; matched pairs {len(m):,}; "
          f"S1 with a match {m.s1.nunique() / n1:.4f}", flush=True)
    for c, g in m.groupby("country"):
        print(f"  {c}: matched pairs {len(g):,}", flush=True)
    val = os.path.join(DATA, "..", "utils", "validate_submission.py")
    subprocess.run(["python3", val, "--matching", os.path.join(a.out, "matching_results.tsv"),
                    "--candidate", os.path.join(a.out, "candidate_pairs.tsv"),
                    "--test-dir", os.path.join(DATA, "test")], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("out")
    ap.add_argument("thr")
    ap.add_argument("unseen_thr", nargs="?")
    ap.add_argument("--cand_p1", type=float, default=1e-4)
    main(ap.parse_args())
