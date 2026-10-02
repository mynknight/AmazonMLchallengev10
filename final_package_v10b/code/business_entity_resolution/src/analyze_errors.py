"""Error analysis on validation scores: where do we lose F0.5?

    python analyze_errors.py <work> <model_name>
"""
import os
import sys

import numpy as np
import pandas as pd

from decide import best_per_query, decide, decide_ef
from metrics import f05, load_gt
from train import val_entities


def main(work, name):
    import json
    sc = pd.read_parquet(os.path.join(work, f"val_scores_{name}.parquet"))
    cfg = json.load(open(os.path.join(work, f"{name}_thresholds.json")))
    ents = val_entities(work)
    gt = load_gt(os.path.join(work, "train_gt.parquet"))
    best = best_per_query(sc)
    if cfg["rule"] == "ef":
        g = cfg["global"]
        pred = decide_ef(sc, best, miss=g["ef_miss"], floor=g["ef_floor"])
    else:
        pred = decide(sc, cfg["global"]["threshold"], best)
    rows = []
    for s, c in zip(ents.entity_id, ents.country):
        t, p = gt[s], pred.get(s, set())
        rows.append((s, c, len(t), len(p), len(t & p), f05(p, t)))
    r = pd.DataFrame(rows, columns=["s1", "country", "n_true", "n_pred", "tp", "f"])
    print("macro F0.5 by country:\n", r.groupby("country").f.mean().round(4).to_string())
    r["kind"] = np.select(
        [(r.n_true == 0) & (r.n_pred == 0), (r.n_true == 0), (r.n_pred == 0), (r.tp == r.n_pred) & (r.tp == r.n_true),
         (r.tp == r.n_pred)],
        ["singleton_ok", "singleton_FP", "missed_all", "perfect", "precise_partial_recall"], "has_FP")
    agg = r.groupby("kind").agg(n=("f", "size"), mean_f=("f", "mean"))
    agg["lost"] = (agg.n * (1 - agg.mean_f)) / len(r)
    print("\nentity outcome buckets (lost = share of total macro-F lost):\n", agg.round(4).to_string())
    # where do the false-negative pairs sit?  (not in candidates / not best for query / below threshold)
    ex = pd.DataFrame([(s, q) for s in ents.entity_id for q in gt[s]], columns=["s1", "qid"])
    cand = sc[sc.label == 1][["qid", "s1", "prob"]]
    ex = ex.merge(cand, on=["qid", "s1"], how="left")
    bq = best.set_index("qid")
    ex["best_s1"] = bq.s1.reindex(ex.qid).values
    ex["best_p"] = bq.prob.reindex(ex.qid).values
    ex["predicted"] = [q in pred.get(s, ()) for s, q in zip(ex.s1, ex.qid)]
    fn = ex[~ex.predicted]
    print(f"\ntrue pairs {len(ex):,}; missed {len(fn):,}")
    print("  not in candidate set :", int(fn.prob.isna().sum()))
    print("  another S1 scored higher:", int(((fn.best_s1 != fn.s1) & fn.prob.notna()).sum()))
    print("  best but below cut      :", int(((fn.best_s1 == fn.s1)).sum()))
    fp = best[(best.label == 0)]
    fp = fp[[q in pred.get(s, ()) for s, q in zip(fp.s1, fp.qid)]]
    print(f"false-positive pairs: {len(fp):,}  (of which query truly unmatched: "
          f"{int((~fp.qid.isin(set(ex.qid))).sum()):,})")
    return r, ex, fp


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
