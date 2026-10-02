"""Name-only country-wide retrieval for EMPTY-address queries: recall at several depths kn.

    python empty_address_recall.py <work_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from blocking import block_partition  # noqa: E402
from pipeline import VAL_STATES, _gt_inverse  # noqa: E402
from run_blocking import _read  # noqa: E402

work = sys.argv[1]
q2s1 = _gt_inverse(work)
for country, states in VAL_STATES.items():
    d = _read(work, "train", 1, country)
    q = pd.concat([_read(work, "train", s, country) for s in (2, 3)], ignore_index=True)
    q = q[(q.ad == "") & (q.nums == "")].reset_index(drop=True)
    true_s1 = q2s1.reindex(q.entity_id.values).values
    st = pd.Series(d.state.values, index=d.entity_id.values)
    tgt = pd.notna(true_s1) & np.isin(st.reindex(true_s1).values, list(states))
    q, true_s1 = q[tgt].reset_index(drop=True), true_s1[tgt]
    print(f"{country}: {len(q):,} empty-address queries whose true S1 is in a validation state; docs {len(d):,}")
    for kn in (5, 10, 20, 40):
        qi, di, cn, ca = block_partition(d, q, max_df=20000, kn=kn, ka=1, kc=1)
        hit = np.zeros(len(q), bool)
        hit[qi[d.entity_id.values[di] == true_s1[qi]]] = True
        print(f"  kn={kn:>3}: recall {hit.mean():.4f}   pairs/query {len(qi) / len(q):.1f}", flush=True)
