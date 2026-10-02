"""Detailed validation error analysis with examples (raw names / addresses).

    python errors_detail.py <work_dir> <model_name>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from decide import best_per_query, decide, decide_ef  # noqa: E402
from metrics import load_gt  # noqa: E402
from train import val_entities  # noqa: E402

pd.set_option("display.width", 250)
work, name = sys.argv[1], sys.argv[2]
sc = pd.read_parquet(os.path.join(work, f"val_scores_{name}.parquet"))
cfg = json.load(open(os.path.join(work, f"{name}_thresholds.json")))
ents = val_entities(work)
gt = load_gt(os.path.join(work, "train_gt.parquet"))
best = best_per_query(sc)
g = cfg["global"]
pred = decide_ef(sc, best, miss=g["ef_miss"], floor=g["ef_floor"]) if cfg["rule"] == "ef" else decide(sc, g["threshold"], best)
cut = g["threshold"]

cols = ["entity_id", "business_name", "business_address"]
raw = pd.concat([pd.read_parquet(os.path.join(work, f"train_s{s}.parquet"), columns=cols) for s in (1, 2, 3)])
raw = raw.set_index("entity_id")
nrm = pd.concat([pd.read_parquet(os.path.join(work, f"train_s{s}_norm.parquet"), columns=["entity_id", "state", "core", "nums"])
                 for s in (1, 2, 3)]).set_index("entity_id")

ex = pd.DataFrame([(s, q) for s in ents.entity_id for q in gt[s]], columns=["s1", "qid"])
ex = ex.merge(sc[sc.label == 1][["qid", "s1", "prob"]], on=["qid", "s1"], how="left")
bq = best.set_index("qid")
ex["best_s1"] = bq.s1.reindex(ex.qid).values
ex["best_p"] = bq.prob.reindex(ex.qid).values
ex["predicted"] = [q in pred.get(s, ()) for s, q in zip(ex.s1, ex.qid)]
ex["q_state"] = nrm.state.reindex(ex.qid).values
ex["s1_state"] = nrm.state.reindex(ex.s1).values
fn = ex[~ex.predicted].copy()
fn["kind"] = np.select([fn.prob.isna(), fn.best_s1 != fn.s1], ["not_candidate", "other_s1_higher"], "below_cut")


def show(d, n, title, extra=()):
    print(f"\n--- {title}: {len(d):,} (showing {min(n, len(d))})")
    for r in d.sample(min(n, len(d)), random_state=0).itertuples():
        s1n, s1a = raw.loc[r.s1, "business_name"], raw.loc[r.s1, "business_address"]
        qn, qa = raw.loc[r.qid, "business_name"], raw.loc[r.qid, "business_address"]
        info = " ".join(f"{k}={getattr(r, k):.3f}" if isinstance(getattr(r, k), float) else f"{k}={getattr(r, k)}" for k in extra)
        print(f"  S1 {s1n!r:42.42} | {s1a!r:70.70}\n  {r.qid[:2]} {qn!r:42.42} | {qa!r:70.70}   {info}")


print(f"missed true pairs by kind:\n{fn.kind.value_counts().to_string()}")
nc = fn[fn.kind == "not_candidate"]
st = np.where(nc.q_state == "", "query has no state", np.where(nc.q_state == nc.s1_state, "same state", "different state"))
print("\nnot-in-candidates by state relation:\n", pd.Series(st).value_counts().to_string())
print("\nall true pairs by state relation:\n", pd.Series(np.where(ex.q_state == "", "query has no state",
      np.where(ex.q_state == ex.s1_state, "same state", "different state"))).value_counts().to_string())
bc = fn[fn.kind == "below_cut"]
print("\nbelow-cut true pairs, probability histogram:")
print(pd.cut(bc.prob, [0, .05, .1, .2, .3, .4, .5, .6, .7]).value_counts().sort_index().to_string())
show(nc[st == "same state"], 12, "NOT IN CANDIDATES, same state")
show(nc[st == "different state"], 6, "NOT IN CANDIDATES, different state", ("q_state", "s1_state"))
show(bc[bc.prob > 0.3], 15, "BELOW CUT, 0.3 < p < cut", ("prob",))
show(bc[bc.prob <= 0.3], 15, "BELOW CUT, p <= 0.3", ("prob",))
show(fn[fn.kind == "other_s1_higher"], 8, "ANOTHER S1 SCORED HIGHER", ("prob", "best_p"))

# false positives: predicted pair not in truth
fp = best[best.label == 0].copy()
fp = fp[[q in pred.get(s, ()) for s, q in zip(fp.s1, fp.qid)]]
q2s1 = pd.Series(ex.s1.values, index=ex.qid.values)
fp["true_s1"] = q2s1.reindex(fp.qid).values
fp["same_core"] = nrm.core.reindex(fp.qid).values == nrm.core.reindex(fp.s1).values
n1 = lambda ids: nrm.nums.reindex(ids).str.split().str[0].fillna("").values
fp["same_n1"] = n1(fp.qid) == n1(fp.s1)
print(f"\nfalse positives {len(fp):,}: query truly unmatched {fp.true_s1.isna().mean():.3f}, "
      f"identical core {fp.same_core.mean():.3f}, identical first number {fp.same_n1.mean():.3f}")
print("FP probability histogram:\n", pd.cut(fp.prob, sorted({cut, .8, .9, .95, .98, .99, 1})).value_counts().sort_index().to_string())
show(fp, 25, "FALSE POSITIVES", ("prob",))
