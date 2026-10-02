"""End-to-end partition loop: blocking -> features -> (training rows | model scores).

    python pipeline.py features <work> [train_rate]      # train split -> train/val feature files
    python pipeline.py predict  <work> <split> <model>   # score all candidates of a split

Validation = every S1 entity located in VAL_STATES (a geographic hold-out, so the
validation entities, their queries and their distractors are never trained on).
"""
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from decide import apply_rules
from pairs import build_features
from run_blocking import iter_partitions

VAL_STATES = {"US": {"wa", "ia", "ct"}, "India": {"kl", "od"}}


def _block_kw(default=dict(max_df=5000, kn=3, ka=3, kc=6)):
    """Blocking settings; override with e.g. ER_BLOCK_KW="max_df=20000,kn=5,ka=5,kc=10"."""
    kw = dict(default)
    for item in filter(None, os.environ.get("ER_BLOCK_KW", "").split(",")):
        k, v = item.split("=")
        kw[k.strip()] = int(v)
    return kw


BLOCK_KW = _block_kw()


S3_FLAG = 1 << 40


def encode_ids(ids):
    """'S2-123' -> 123, 'S3-123' -> 123 | S3_FLAG, 'S1-123' -> 123 (int64)."""
    s = pd.Series(ids)
    num = s.str.slice(3).astype(np.int64).values
    return np.where(s.str.startswith("S3").values, num | S3_FLAG, num)


def decode_ids(x, prefix=None):
    """Turn packed int64 ids back into 'S2-<n>' / 'S3-<n>' strings (S3 flagged by a high bit), or '<prefix><n>' when a prefix is given."""
    x = np.asarray(x, np.int64)
    if prefix is not None:
        return np.char.add(prefix, x.astype(str))
    s3 = (x & S3_FLAG) != 0
    return np.where(s3, np.char.add("S3-", (x & ~S3_FLAG).astype(str)), np.char.add("S2-", x.astype(str)))


def _hash01(ids):
    """Deterministic pseudo-random number in [0, 1] per id (MD5-based), used for reproducible sampling and folds."""
    return np.array([int(hashlib.md5(x.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for x in ids])


def _gt_inverse(work):
    """Invert the train ground truth into a Series query_id -> its true S1 id."""
    gt = pd.read_parquet(os.path.join(work, "train_gt.parquet"))
    ex = gt.assign(q=gt.matched_entity_ids.str.split(",")).explode("q")
    ex = ex[ex.q.str.len() > 0]
    return pd.Series(ex.source1_entity_id.values, index=ex.q.values)


class _Writer:
    """Incremental parquet writer that creates the file with the schema of the first non-empty frame."""
    def __init__(self, path):
        """Remember the output path; the writer is opened lazily on the first write."""
        self.path, self.w = path, None

    def write(self, df):
        """Append a frame (cast to the file schema); empty frames are skipped."""
        if len(df) == 0:
            return
        t = pa.Table.from_pandas(df, preserve_index=False)
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, t.schema)
        self.w.write_table(t.cast(self.w.schema))

    def close(self):
        """Close the parquet file if anything was written."""
        if self.w is not None:
            self.w.close()


def make_features(work, train_rate=0.2, **kw):
    """Block every train partition and write pair features: validation-state queries to feat_val, a hashed train_rate share of the other queries to feat_train (whole queries, with labels)."""
    q2s1 = _gt_inverse(work)
    wt = _Writer(os.path.join(work, "feat_train.parquet"))
    wv = _Writer(os.path.join(work, "feat_val.parquet"))
    for info, d, q, qi, di, cn, ca in iter_partitions(work, "train", **BLOCK_KW, **kw):
        t0 = time.time()
        val_states = VAL_STATES.get(info["country"], set())
        is_val = np.isin(d.state.values[di], list(val_states))
        h = _hash01(q.entity_id.values)
        is_tr = (~is_val) & (h[qi] < train_rate)
        keep = is_val | is_tr
        if not keep.any():
            continue
        # context features must see the query's full candidate list: keep whole queries
        kq = np.zeros(len(q), bool); kq[qi[keep]] = True
        sel = kq[qi]
        c = build_features(d, q, qi, di, cn, ca, sel=sel)
        c["part"] = np.float32(info["part"])
        c["label"] = (q2s1.reindex(c.qid.values).values == c.s1.values).astype(np.int8)
        c["is_val"] = is_val[sel]
        c["country"] = info["country"]
        wv.write(c[c.is_val].drop(columns=["is_val"]))
        wt.write(c[~c.is_val].drop(columns=["is_val"]))
        print("  {country}/{state}: q={n_q:,} pairs={n_pairs:,} block {secs:.0f}s".format(**info),
              f"feat rows={len(c):,} pos={c.label.sum():,} {time.time() - t0:.0f}s", flush=True)
    wt.close(); wv.close()


def predict(work, split, model_path, unseen_model_path=None, out_name=None, **kw):
    """Score every candidate pair of `split`.  Countries that do not occur in the training data
    (France) are scored with `unseen_model_path` when given: a model without the vocabulary-
    dependent extra-word features, whose word table is learned on US / India names and is blank
    for another language (on validation, blanking those features costs up to 0.06 F0.5)."""
    import json
    import lightgbm as lgb
    from run_blocking import countries_of
    models = {"seen": lgb.Booster(model_file=model_path)}
    if unseen_model_path:
        models["unseen"] = lgb.Booster(model_file=unseen_model_path)
    seen = set(countries_of(work, "train"))
    table = None
    if any(f.startswith("w_") for m in models.values() for f in m.feature_name()):
        with open(os.path.join(work, "extra_words.json"), encoding="utf-8") as fh:  # from augment.py
            table = json.load(fh)
    w = _Writer(os.path.join(work, out_name or f"{split}_scores.parquet"))
    for info, d, q, qi, di, cn, ca in iter_partitions(work, split, **BLOCK_KW, **kw):
        t0 = time.time()
        which = "unseen" if (info["country"] not in seen and "unseen" in models) else "seen"
        booster = models[which]
        feats = booster.feature_name()
        uses_words = any(f.startswith("w_") for f in feats)
        c = build_features(d, q, qi, di, cn, ca, extra_table=table if uses_words else None)
        c["part"] = np.float32(info["part"])
        prob = apply_rules(c, booster.predict(c[feats].values.astype(np.float32),
                                              num_threads=int(os.environ.get("ER_THREADS", os.cpu_count()))))
        out = pd.DataFrame({"q": encode_ids(c.qid.values), "s1": encode_ids(c.s1.values),
                            "prob": prob.astype(np.float32)})
        out["country"] = info["country"]
        w.write(out)
        print("  {country}/{state}: q={n_q:,} pairs={n_pairs:,} block {secs:.0f}s".format(**info),
              f"scored with {which} model {time.time() - t0:.0f}s", flush=True)
    w.close()


if __name__ == "__main__":
    mode, work = sys.argv[1], sys.argv[2]
    if mode == "features":
        make_features(work, float(sys.argv[3]) if len(sys.argv) > 3 else 0.2)
    elif mode == "predict":  # predict <work> <split> <model> [<model for countries unseen in training>]
        predict(work, sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
