"""France proxy: blocking recall of in-state search vs country-wide search on the validation states.

France has no parsed state, so it is always searched country-wide.  This measures how much
recall that costs, using US / India validation states where labels exist.

    python blocking_countrywide.py <work_dir>
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from blocking import block_partition  # noqa: E402
from pipeline import BLOCK_KW, VAL_STATES, _gt_inverse  # noqa: E402
from run_blocking import _read  # noqa: E402

work = sys.argv[1]
q2s1 = _gt_inverse(work)
print("BLOCK_KW", BLOCK_KW)
rows = []
for country, states in VAL_STATES.items():
    dall = _read(work, "train", 1, country)
    for st in sorted(states):
        q = pd.concat([_read(work, "train", s, country, st) for s in (2, 3)], ignore_index=True)
        d = dall[dall.state == st].reset_index(drop=True)
        true_s1 = q2s1.reindex(q.entity_id.values).values
        target = np.isin(true_s1, d.entity_id.values)          # true S1 lives in this state
        for mode, dd in (("in-state", d), ("country-wide", dall)):
            t0 = time.time()
            qi, di, cn, ca = block_partition(dd, q, **BLOCK_KW)
            hit = np.zeros(len(q), bool)
            hit[qi[dd.entity_id.values[di] == true_s1[qi]]] = True
            rows.append(dict(country=country, state=st, mode=mode, docs=len(dd), queries=int(target.sum()),
                             recall=hit[target].mean(), pairs_per_q=len(qi) / len(q), secs=time.time() - t0))
            print(rows[-1], flush=True)
print(pd.DataFrame(rows).round(4).to_string(index=False))
