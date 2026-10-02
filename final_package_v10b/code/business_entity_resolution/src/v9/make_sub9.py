"""Write the submission TSVs: US/India from the stage-C matcher, France from a France score file.

    python make_sub9.py <out_dir> --usin SCORES --usin_thr THR.json --fr new --fr_new SCORES --fr_thr T

US/India: best S1 per query, kept if prob >= the per-country threshold of --usin_thr.
France:   new = the decisions of --fr_new (best S1 per query, prob >= --fr_thr); this is what the
          final submission uses (stage D writes --fr_new).  The stage-B scores
          (test_scores_fin_e5tr.parquet) supply France's candidate pairs; the policies v3 / add are
          kept from development and apply the stage-B France score at --fr_v3_thr.
candidate_pairs.tsv = every pair scored by the final models (stage-1 p1 >= 0.01).
"""
import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from common9 import CE_WORK, DATA
from io_utils import read_tsv


def best(d, thr):
    b = d.sort_values(["qid", "prob"], ascending=[True, False]).drop_duplicates("qid")
    return b[b.prob.values >= thr(b)]


def write_lists(path, header, s1_ids, s1, ids):
    grp = pd.Series(ids).groupby(s1).agg(lambda x: ",".join(dict.fromkeys(x)))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for sid in s1_ids:
            f.write(f"{sid}\t{grp.get(sid, '')}\n")


def main(a):
    v5a = pd.read_parquet(os.path.join(CE_WORK, "test_scores_fin_e5tr.parquet"))
    v5a = v5a[v5a.p1 >= 0.01]
    usin = pd.read_parquet(a.usin) if a.usin else v5a[v5a.country != "France"]
    usin = usin[usin.country != "France"]
    thr = json.load(open(a.usin_thr))
    m_us = best(usin, lambda b: b.country.map(lambda c: thr[c]["threshold"]).values)
    fr_old = v5a[v5a.country == "France"]
    m_old = best(fr_old, lambda b: a.fr_v3_thr)
    if a.fr == "v3":
        m_fr = m_old
    else:
        fr_new = pd.read_parquet(a.fr_new)
        m_new = best(fr_new, lambda b: a.fr_thr)
        m_fr = m_new if a.fr == "new" else pd.concat([m_old, m_new[~m_new.qid.isin(fr_old_best_q(fr_old))]])
    m = pd.concat([m_us[["qid", "s1", "prob"]], m_fr[["qid", "s1", "prob"]]])
    assert not m.qid.duplicated().any()
    cand = pd.concat([usin[["qid", "s1"]], fr_old[["qid", "s1"]]]).drop_duplicates()
    assert len(m.merge(cand, on=["qid", "s1"])) == len(m), "match outside the candidate set"
    s1_ids = read_tsv(os.path.join(DATA, "test", "test_source1.tsv"), usecols=["entity_id"]).entity_id.values
    os.makedirs(a.out, exist_ok=True)
    cand = cand.sort_values(["s1", "qid"])
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), "source1_entity_id\tcandidate_entity_ids",
                s1_ids, cand.s1.values, cand.qid.values)
    m = m.sort_values(["s1", "prob"], ascending=[True, False])
    write_lists(os.path.join(a.out, "matching_results.tsv"), "source1_entity_id\tmatched_entity_ids",
                s1_ids, m.s1.values, m.qid.values)
    print(f"candidates {len(cand):,}; matched pairs {len(m):,} (US/India {len(m_us):,}, France {len(m_fr):,} "
          f"vs v3 {len(m_old):,})", flush=True)
    val = os.path.join(DATA, "..", "utils", "validate_submission.py")
    subprocess.run(["python3", val, "--matching", os.path.join(a.out, "matching_results.tsv"),
                    "--candidate", os.path.join(a.out, "candidate_pairs.tsv"),
                    "--test-dir", os.path.join(DATA, "test"), "--check-ids"], check=False)


def fr_old_best_q(fr_old):
    """Queries that v3 matched (their v3 decision is kept in the 'add' policy)."""
    return best(fr_old, lambda b: 0.79).qid.values


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--usin", default="")
    ap.add_argument("--usin_thr", default=os.path.join(CE_WORK, "fin_tr_thresholds.json"))
    ap.add_argument("--fr", default="v3", choices=["v3", "add", "new"])
    ap.add_argument("--fr_new", default="")
    ap.add_argument("--fr_thr", type=float, default=0.79)
    ap.add_argument("--fr_v3_thr", type=float, default=0.79, help="threshold of the v3 France model (V7: 0.79)")
    main(ap.parse_args())
