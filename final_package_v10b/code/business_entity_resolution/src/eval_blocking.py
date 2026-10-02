"""Blocking quality on train: pair recall, per-entity recall ceiling, candidate volume."""
import os
import sys

import numpy as np
import pandas as pd

from metrics import f05


def main(work, cands_file="train_cands.parquet", countries=None):
    c = pd.read_parquet(os.path.join(work, cands_file), columns=["qid", "s1"])
    gt = pd.read_parquet(os.path.join(work, "train_gt.parquet"))
    s1c = pd.read_parquet(os.path.join(work, "train_s1.parquet"), columns=["entity_id", "country"])
    if countries:
        keep = set(s1c.entity_id[s1c.country.isin(countries)])
        gt = gt[gt.source1_entity_id.isin(keep)]
    ex = gt.assign(qid=gt.matched_entity_ids.str.split(",")).explode("qid")
    ex = ex[ex.qid.str.len() > 0][["source1_entity_id", "qid"]].rename(columns={"source1_entity_id": "s1"})
    hit = ex.merge(c, on=["qid", "s1"], how="left", indicator=True)
    hit["found"] = hit["_merge"] == "both"
    print(f"pairs: {len(c):,}  true pairs: {len(ex):,}  pair recall: {hit.found.mean():.4f}")
    per = hit.groupby("s1").found.agg(["sum", "count"])
    ceil = per.apply(lambda r: f05({1} if r["sum"] else set(), {1}) if r["sum"] == 0 else
                     (1.25 * r["sum"] / r["count"]) / (0.25 + r["sum"] / r["count"]), axis=1)
    n_single = (gt.matched_entity_ids.str.len() == 0).sum()
    macro = (ceil.sum() + n_single) / len(gt)
    print(f"macro F0.5 ceiling (perfect matcher on candidates): {macro:.4f}")
    q_per = c.groupby("qid").size()
    print(f"candidates per query: mean {q_per.mean():.2f}, queries covered {len(q_per):,}")
    return hit


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "train_cands.parquet",
         sys.argv[3].split(",") if len(sys.argv) > 3 else None)
