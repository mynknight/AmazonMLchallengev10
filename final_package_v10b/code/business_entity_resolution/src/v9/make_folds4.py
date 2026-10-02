"""pairs_train.parquet with 4 folds by the source-1 (country, state), greedy balance per country
(same principle as crossencoder/src/ce_data._folds, K = 4)."""
import os
import numpy as np
import pandas as pd
from common9 import CE_WORK, V2, WORK9

K = 4
tr = pd.read_parquet(os.path.join(CE_WORK, "pairs_train.parquet"))
st = pd.read_parquet(os.path.join(V2, "train_s1_norm.parquet"), columns=["entity_id", "state"]).set_index("entity_id").state
key = tr.country.astype(str) + "/" + st.reindex(tr.s1.values).fillna("").replace("", "none").values
fold, load = {}, {}
for k, n in key.value_counts().items():
    c = k.split("/")[0]
    f = int(np.argmin([load.get((c, j), 0) for j in range(K)]))
    fold[k] = f
    load[(c, f)] = load.get((c, f), 0) + n
tr["fold2"] = tr.fold
tr["fold"] = key.map(fold).values.astype(np.int8)
tr.to_parquet(os.path.join(WORK9, "ce4", "pairs_train.parquet"), index=False)
print(pd.crosstab(tr.country, tr.fold))
