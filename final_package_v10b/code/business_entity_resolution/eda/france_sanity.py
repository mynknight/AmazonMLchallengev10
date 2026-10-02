"""HANDOFF §5.8: France has no labels -- compare its test predictions with US / India.

    python france_sanity.py <work_dir> <thresholds.json>

Per country: share of records assigned to an entity, share of entities with >= 1 match, the
distribution of matches per entity and of each record's best probability.  If the model
transfers, France should look like US / India (same data generator).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

work, thr_path = sys.argv[1], sys.argv[2]
thr = json.load(open(thr_path))
n_s1 = pd.read_parquet(os.path.join(work, "test_s1.parquet"), columns=["country"]).country.value_counts()
dset = ds.dataset(os.path.join(work, "test_scores.parquet"))
rows, dist, hist = [], {}, {}
for c in sorted(n_s1.index):
    t = dset.to_table(filter=ds.field("country") == c, columns=["q", "s1", "prob"]).to_pandas()
    b = t.sort_values(["q", "prob"], ascending=[True, False]).drop_duplicates("q")
    th = thr.get(c, thr["global"])["threshold"]
    m = b[b.prob >= th]
    per = m.groupby("s1").size()
    counts = np.bincount(per.values, minlength=12)[:12].astype(float)
    counts[0] = n_s1[c] - len(per)
    dist[c] = counts / n_s1[c]
    hist[c] = np.histogram(b.prob, bins=[0, .05, .2, .5, .8, .95, 1.0001])[0] / len(b)
    rows.append(dict(country=c, threshold=th, s1=n_s1[c], records=len(b), matched_records=len(m) / len(b),
                     s1_with_match=len(per) / n_s1[c], matches_per_matched_s1=per.mean(),
                     pairs_per_record=len(t) / len(b)))
print(pd.DataFrame(rows).set_index("country").round(4).to_string())
print("\nmatches per entity (share of entities):")
print(pd.DataFrame(dist, index=[str(i) for i in range(12)]).T.round(4).to_string())
print("\nbest probability per record (share of records):")
print(pd.DataFrame(hist, index=["<.05", ".05-.2", ".2-.5", ".5-.8", ".8-.95", ">=.95"]).T.round(4).to_string())
