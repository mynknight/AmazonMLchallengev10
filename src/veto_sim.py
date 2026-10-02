"""Does a cross-encoder VETO on top of a no-CE model help?  Measured on US/India validation, where
labels exist: the no-CE word-role model (nv_u_noce, same type as v6a's France model fin_nv_u) makes
the decisions; a pair it accepts is dropped when the CE probability is below v.
Test-like macro F0.5 with thresholds tuned per held-out state (t for the model, v for the veto)."""
import os
import numpy as np, pandas as pd
from common10 import CE4, CE_WORK, V2, WORK9
from metrics import load_gt
from train import testlike, val_entities

gt = load_gt(os.path.join(V2, "train_gt.parquet")); ents = val_entities(V2)
st = pd.read_parquet(os.path.join(V2, "train_s1_norm.parquet"), columns=["entity_id", "state"]).set_index("entity_id").state
ents["st"] = st.reindex(ents.entity_id.values).values
ntrue = ents.entity_id.map(lambda s: len(gt[s])).values; eidx = pd.Index(ents.entity_id)
nv = pd.read_parquet(os.path.join(WORK9, "val_scores_nv_u_noce.parquet"))
pv = pd.read_parquet(os.path.join(CE_WORK, "pairs_val.parquet"), columns=["qid", "s1"])
pv["ce"] = np.mean([np.load(os.path.join(CE4, f"ce_e5k4_f{k}_val.npy")) for k in range(4)], axis=0)
d = nv.merge(pv, on=["qid", "s1"], how="left", validate="one_to_one")
d["pce"] = 1 / (1 + np.exp(-d.ce.values))
tl = testlike(d, V2)
best = tl.sort_values(["qid", "prob"], ascending=[True, False]).drop_duplicates("qid")

def perent(b):
    b = b[eidx.get_indexer(b.s1) >= 0]; e = eidx.get_indexer(b.s1)
    npred = np.bincount(e, minlength=len(ents)); tp = np.bincount(e, weights=b.label.values, minlength=len(ents))
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0); r = np.divide(tp, ntrue, out=np.zeros_like(tp), where=ntrue > 0)
    f = np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)

T = np.round(np.arange(0.70, 0.96, 0.02), 2); VV = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
F = {(t, v): perent(best[(best.prob.values >= t) & (best.pce.values >= v)]) for t in T for v in VV}
res = {"no-CE model": 0.0, "no-CE model + CE veto": 0.0, "veto only (model threshold fixed)": 0.0}; picks = []
for c in ["US", "India"]:
    mc = (ents.country == c).values
    for s in sorted(ents.st[mc].unique()):
        te = mc & (ents.st == s).values; trn = mc & ~(ents.st == s).values
        t0 = max(T, key=lambda t: F[(t, 0.0)][trn].mean())
        tv = max(F, key=lambda k: F[k][trn].mean())
        res["no-CE model"] += F[(t0, 0.0)][te].sum(); res["no-CE model + CE veto"] += F[tv][te].sum()
        vf = max(VV, key=lambda v: F[(t0, v)][trn].mean())
        res["veto only (model threshold fixed)"] += F[(t0, vf)][te].sum()
        picks.append(f"{s}: t={tv[0]} veto<{tv[1]} (vs t={t0})")
for k in res: print(f"{k:25} held-out test-like F0.5 {res[k] / len(ents):.5f}")
print("chosen:", "; ".join(picks))
b = best[best.prob >= 0.82]
for v in (0.01, 0.05, 0.2):
    vt = b[b.pce < v]
    print(f"veto < {v}: drops {len(vt):,} accepted pairs ({len(vt) / len(b):.2%}); of them true {vt.label.mean():.3f}")
