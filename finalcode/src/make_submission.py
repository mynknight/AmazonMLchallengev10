"""Write output/matching_results.tsv and output/candidate_pairs.tsv from test scores.

    python make_submission.py <work> <data_dir> <out_dir> <thresholds.json> [<unseen_thresholds.json>]

Decision rule (per country threshold t, 'global' for countries unseen in training):
each S2/S3 record is assigned to its highest-probability S1 candidate if prob >= t.  Countries
unseen in training use the 'global' entry of <unseen_thresholds.json> when given (the
thresholds of the model that scored them, see pipeline.predict).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from decide import expected_f_select
from io_utils import read_tsv
from pipeline import decode_ids


def assign(q, s1, prob, cfg):
    """Best S1 per query, then threshold or expected-F0.5 selection (see decide.py)."""
    order = np.lexsort((-prob, q))
    q, s1, prob = q[order], s1[order], prob[order]
    first = np.r_[True, q[1:] != q[:-1]]
    q, s1, prob = q[first], s1[first], prob[first]
    if cfg.get("rule") == "ef":
        keep = expected_f_select(s1, q, prob.astype(np.float64), miss=cfg["ef_miss"], floor=cfg["ef_floor"])
    else:
        keep = prob >= cfg["threshold"]
    return s1[keep], q[keep]


def iter_groups(s1, q, presorted=False):
    """Yield (s1_int, comma-joined id string) per S1 entity."""
    if not presorted:
        order = np.argsort(s1, kind="stable")
        s1, q = s1[order], q[order]
    if len(s1) == 0:
        return
    ids = decode_ids(q)
    bounds = np.r_[0, np.nonzero(s1[1:] != s1[:-1])[0] + 1, len(s1)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(s1[a]), ",".join(ids[a:b])


def main(work, data_dir, out_dir, thr_path, unseen_thr_path=None, scores="test_scores.parquet"):
    """Stream the scored test pairs per country, apply the per-country thresholds (unseen countries use the second file) and write candidate_pairs.tsv and matching_results.tsv."""
    from run_blocking import countries_of
    os.makedirs(out_dir, exist_ok=True)
    with open(thr_path) as f:
        thr = json.load(f)
    unseen_thr = None
    if unseen_thr_path:
        with open(unseen_thr_path) as f:
            unseen_thr = json.load(f)
    seen = set(countries_of(work, "train"))
    s1 = read_tsv(os.path.join(data_dir, "test", "test_source1.tsv"), usecols=["entity_id"])
    s1_ids = s1.entity_id.values
    s1_int = s1.entity_id.str.slice(3).astype(np.int64).values
    dset = ds.dataset(os.path.join(work, scores))
    countries = sorted(set(dset.to_table(columns=["country"]).column("country").unique().to_pylist()))
    matches = {}
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    cf = open(cand_path, "w", encoding="utf-8", newline="\n")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")
    written = set()
    for c in countries:
        src = unseen_thr if (unseen_thr is not None and c not in seen) else thr
        cfg = dict(src.get(c, src["global"]))
        cfg.setdefault("rule", src.get("rule", "threshold"))
        tab = dset.to_table(filter=ds.field("country") == c, columns=["q", "s1", "prob"])
        q = tab.column("q").to_numpy(); s = tab.column("s1").to_numpy(); p = tab.column("prob").to_numpy()
        del tab
        ms, mq = assign(q, s, p, cfg)
        matches.update(iter_groups(ms, mq))
        # candidate set = every pair the model scored (dedup q within an S1)
        pairs = np.unique(np.stack([s, q], 1), axis=0)
        for si, txt in iter_groups(pairs[:, 0], pairs[:, 1], presorted=True):
            cf.write(f"S1-{si}\t{txt}\n")
            written.add(si)
        print(f"{c}: rule {cfg['rule']} t={cfg.get('threshold')}, pairs {len(q):,}, matched records {len(mq):,}, "
              f"S1 with matches {len(set(ms.tolist())):,}", flush=True)
        del q, s, p, pairs
    with open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, si in zip(s1_ids, s1_int):
            f.write(f"{sid}\t{matches.get(int(si), '')}\n")
    for sid, si in zip(s1_ids, s1_int):
        if int(si) not in written:
            cf.write(f"{sid}\t\n")
    cf.close()
    print("written", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:6])
