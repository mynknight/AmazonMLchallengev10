"""Compare two models on the SAME validation entities (the intersection of their validation sets;
normalisation fixes can move entities in or out of the validation states).

    python compare_val.py <workA> <modelA> <workB> <modelB>

Per country: exact macro F0.5 with the best per-country threshold, standard and test-like.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from decide import tune_fine  # noqa: E402
from metrics import load_gt  # noqa: E402
from train import testlike, val_entities  # noqa: E402

wa, ma, wb, mb = sys.argv[1:5]
ea, eb = val_entities(wa), val_entities(wb)
common = ea.merge(eb[["entity_id"]], on="entity_id")
print(f"validation entities: A {len(ea):,}, B {len(eb):,}, common {len(common):,}")
gt = load_gt(os.path.join(wa, "train_gt.parquet"))
for work, name in ((wa, ma), (wb, mb)):
    sc = pd.read_parquet(os.path.join(work, f"val_scores_{name}.parquet"))
    out = []
    for tag, data in (("std", sc), ("test-like", testlike(sc, work))):
        for country, g in list(common.groupby("country")):
            ids = g.entity_id.tolist()
            t, f = tune_fine(data[data.country == country], {s: gt[s] for s in ids}, ids)
            out.append((tag, country, len(ids), t, f))
    r = pd.DataFrame(out, columns=["eval", "country", "n", "thr", "f05"])
    # entity-weighted global value from the per-country optima (as the submission uses them)
    glob = r.groupby("eval").apply(lambda d: np.average(d.f05, weights=d.n)).round(4).to_dict()
    print(f"{name}: " + " | ".join(f"{e}/{c} {f:.4f}@{t:.2f}" for e, c, n, t, f in out) + f" | global {glob}")
