"""France rows of the stage-B test scores (v6a: fin_nv_u, label-free word roles, no CE)
-> <ER_V10_WORK>/test_fr_v6a.parquet (qid, s1, country, p1, prob; stage-1 p1 >= 0.01)."""
import os

import pandas as pd

from common10 import CE_WORK, WORK10

t = pd.read_parquet(os.path.join(CE_WORK, "test_scores_fin_e5tr.parquet"))
fr = t[(t.country == "France") & (t.p1 >= 0.01)].reset_index(drop=True)
fr.to_parquet(os.path.join(WORK10, "test_fr_v6a.parquet"), index=False)
print(f"France test pairs: {len(fr):,}")
