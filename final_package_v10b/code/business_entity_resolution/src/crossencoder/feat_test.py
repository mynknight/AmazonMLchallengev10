"""Re-run the v2 test pipeline (blocking + 70 features + stage-1 LightGBM) and keep the features of
the pairs that pass the stage-1 filter (p1 >= P1_MIN), so a final model can use them next to the
cross-encoder scores.  The v2 run only saved probabilities.

    ER_BLOCK_KW="max_df=20000,kn=5,ka=5,kc=10" python feat_test.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from common import P1_MIN, THREADS, V2, WORK
from decide import apply_rules
from pairs import build_features
from pipeline import BLOCK_KW, _Writer
from run_blocking import countries_of, iter_partitions


def main():
    import lightgbm as lgb
    assert BLOCK_KW == dict(max_df=20000, kn=5, ka=5, kc=10), f"v2 blocking settings expected, got {BLOCK_KW}"
    models = {"seen": lgb.Booster(model_file=os.path.join(V2, "lgb_v2.txt")),
              "unseen": lgb.Booster(model_file=os.path.join(V2, "lgb_v2_novocab.txt"))}
    seen = set(countries_of(V2, "train"))
    with open(os.path.join(V2, "extra_words.json"), encoding="utf-8") as fh:
        table = json.load(fh)
    w = _Writer(os.path.join(WORK, "feat_test_p1.parquet"))
    tot = 0
    for info, d, q, qi, di, cn, ca in iter_partitions(V2, "test", **BLOCK_KW):
        t0 = time.time()
        booster = models["seen" if info["country"] in seen else "unseen"]
        c = build_features(d, q, qi, di, cn, ca, extra_table=table)
        c["part"] = np.float32(info["part"])
        p1 = apply_rules(c, booster.predict(c[booster.feature_name()].values.astype(np.float32),
                                            num_threads=THREADS))
        keep = p1 >= P1_MIN
        c = c[keep].reset_index(drop=True)
        c["p1"] = p1[keep].astype(np.float32)
        c["country"] = info["country"]
        w.write(c)
        tot += len(c)
        print("  {country}/{state}: q={n_q:,} pairs={n_pairs:,}".format(**info),
              f"kept {len(c):,} (total {tot:,}) {time.time() - t0:.0f}s", flush=True)
    w.close()


if __name__ == "__main__":
    main()
