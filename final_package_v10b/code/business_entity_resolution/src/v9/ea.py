"""Deeper candidate retrieval for empty-address queries ("ea"), through the whole V9 cascade.

    python ea.py block <train|test>   # re-block the country-wide fallback partitions, v2 features + stage-1 p1
    python ea.py ce <train|val|test>  # 4-fold e5k4 cross-encoder scores of the new rows (GPU)
    python ea.py u <train|val|test>   # u_* features of the new rows

Empty-address S2/S3 records have no state and are searched country-wide.  The v2 blocker keeps
top-5 / top-10 by name and skips name 3-grams with df > 20000, which on validation retrieves only
72% (US) / 68% (India) of their true pairs.  EA_KW (max_df=100000, kn=40, kc=40) retrieves 87.6% (US).

For each fallback partition, the empty-address queries are blocked with EA_KW and the other fallback
queries with the v2 setting; features are built on the union so the query / S1 context features see
the whole partition (as in pipeline.make_features).  Only rows of empty-address queries are kept:
they REPLACE the old rows of those queries in the train / val / test sets.  Rows are stage-1 filtered
at p1 >= 1e-4 like crossencoder/work/pairs_*.parquet.  France is left unchanged.
Writes work/ea/rows_<split>.parquet (+ ce_<split>.npy, u_<split>.parquet), row-aligned.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from common9 import CE_WORK, THREADS, V2, WORK9
from blocking import block_partition
from decide import apply_rules
from pairs import build_features
from pipeline import VAL_STATES, _gt_inverse, _hash01
from run_blocking import _read, _states

EA = os.environ.get("EA_DIR", os.path.join(WORK9, "ea"))
EA_KW = dict(max_df=100000, kn=40, ka=5, kc=40)
V2_KW = dict(max_df=20000, kn=5, ka=5, kc=10)
P1_MIN = 1e-4
TRAIN_RATE = float(os.environ.get("EA_RATE", 0.3))  # 1.0 = every training query (full_train.py)


def fallback(split, country):
    s1_states = _states(V2, split, 1, country)
    d = _read(V2, split, 1, country)
    q = pd.concat([_read(V2, split, s, country, exclude_states=s1_states) for s in (2, 3)], ignore_index=True)
    return d, q


def block(split):
    import lightgbm as lgb
    os.makedirs(EA, exist_ok=True)
    b = lgb.Booster(model_file=os.path.join(V2, "lgb_v2.txt"))
    table = json.load(open(os.path.join(V2, "extra_words.json"), encoding="utf-8"))
    q2s1 = _gt_inverse(V2) if split == "train" else None
    out = []
    for country in ("US", "India"):
        t0 = time.time()
        d, q = fallback(split, country)
        empty = ((q.ad == "") & (q.nums == "")).values
        ie, ine = np.nonzero(empty)[0], np.nonzero(~empty)[0]
        parts = []
        for idx, kw in ((ie, EA_KW), (ine, V2_KW)):
            if len(idx) == 0:
                continue
            qi, di, cn, ca = block_partition(d, q.iloc[idx].reset_index(drop=True), **kw)
            parts.append((idx[qi], di, cn, ca))
        qi, di, cn, ca = (np.concatenate(x) for x in zip(*parts))
        keep = empty[qi]
        if split == "train":  # rows of the training sample (hash < TRAIN_RATE) or of validation S1
            # as pipeline.make_features: keep WHOLE queries that are sampled (hash < TRAIN_RATE) or
            # that touch a validation S1; their non-validation pairs become training rows
            is_val = d.state.isin(VAL_STATES[country]).values[di]
            h = _hash01(q.entity_id.values)
            kq = np.zeros(len(q), bool); kq[qi[is_val | (h[qi] < TRAIN_RATE)]] = True
            keep &= kq[qi]
        c = build_features(d, q, qi, di, cn, ca, sel=keep, extra_table=table)
        c["part"] = np.float32(1)
        c["country"] = country
        if split == "train":
            c["label"] = (q2s1.reindex(c.qid.values).values == c.s1.values).astype(np.int8)
            c["is_val"] = is_val[keep]
        c["p1"] = apply_rules(c, b.predict(c[b.feature_name()].values.astype(np.float32),
                                           num_threads=THREADS)).astype(np.float32)
        c = c[c.p1.values >= P1_MIN].reset_index(drop=True)
        out.append(c)
        print(f"  {split}/{country}: {empty.sum():,} empty-address queries, {len(c):,} rows with p1 >= {P1_MIN:g} "
              f"({time.time() - t0:.0f}s)", flush=True)
    c = pd.concat(out, ignore_index=True)
    if split == "train":
        c[c.is_val].drop(columns=["is_val"]).to_parquet(os.path.join(EA, "rows_val.parquet"), index=False)
        c[~c.is_val].drop(columns=["is_val"]).to_parquet(os.path.join(EA, "rows_train.parquet"), index=False)
    else:
        c.to_parquet(os.path.join(EA, "rows_test.parquet"), index=False)
    print(f"ea block {split} done", flush=True)


def block_fr():
    """France test: deeper retrieval for empty-address queries, scored by the v3 France model
    (vocabulary-free v2 LightGBM, whose probability is the v3 decision score)."""
    import lightgbm as lgb
    os.makedirs(EA, exist_ok=True)
    b = lgb.Booster(model_file=os.path.join(V2, "lgb_v2_novocab.txt"))
    t0 = time.time()
    d, q = fallback("test", "France")
    empty = ((q.ad == "") & (q.nums == "")).values
    ie, ine = np.nonzero(empty)[0], np.nonzero(~empty)[0]
    parts = []
    for idx, kw in ((ie, EA_KW), (ine, V2_KW)):
        if len(idx) == 0:
            continue
        qi, di, cn, ca = block_partition(d, q.iloc[idx].reset_index(drop=True), **kw)
        parts.append((idx[qi], di, cn, ca))
    qi, di, cn, ca = (np.concatenate(x) for x in zip(*parts))
    c = build_features(d, q, qi, di, cn, ca, sel=empty[qi])
    c["part"] = np.float32(1)
    c["country"] = "France"
    c["p1"] = apply_rules(c, b.predict(c[b.feature_name()].values.astype(np.float32), num_threads=THREADS)).astype(np.float32)
    c = c[c.p1.values >= P1_MIN].reset_index(drop=True)
    c[["qid", "s1", "country", "p1"]].to_parquet(os.path.join(EA, "rows_test_fr.parquet"), index=False)
    print(f"  test/France: {empty.sum():,} empty-address queries of {len(q):,} fallback queries, "
          f"{len(c):,} rows with p1 >= {P1_MIN:g} ({time.time() - t0:.0f}s)", flush=True)


def ce(split):
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification
    from ce_model4 import Collator, predict, special_ids
    rows = pd.read_parquet(os.path.join(EA, f"rows_{split}.parquet"), columns=["qid", "s1", "country"])
    cls_id, sep_id, pad_id, mid, tag = special_ids("intfloat/multilingual-e5-small", "tr")
    z = np.load(os.path.join(WORK9, "ce4", f"tok_{'test' if split == 'test' else 'train'}_{tag}.npz"))
    idx = pd.Index(z["entity_id"])
    qr, sr = idx.get_indexer(rows.qid.values), idx.get_indexer(rows.s1.values)
    assert (qr >= 0).all() and (sr >= 0).all()
    col = Collator(torch.from_numpy(z["ids"].astype(np.int32)), z["off"], cls_id, sep_id, pad_id, mid)
    if split == "train":  # out-of-fold: the model that did not train on the S1's (country, state) fold
        pt = pd.read_parquet(os.path.join(WORK9, "ce4", "pairs_train.parquet"), columns=["s1", "country", "fold"])
        st = pd.read_parquet(os.path.join(V2, "train_s1_norm.parquet"), columns=["entity_id", "state"]).set_index("entity_id").state
        key = lambda df: df.country.astype(str).values + "/" + st.reindex(df.s1.values).fillna("").replace("", "none").values
        fmap = pd.Series(pt.fold.values, index=key(pt)).groupby(level=0).first()
        fold = fmap.reindex(key(rows)).values
        # S1 states absent from the CE training pairs: no fold model saw them, all 4 are out-of-fold
        print(f"  rows whose S1 state is in no CE fold: {np.isnan(fold).sum():,}", flush=True)
        fold = np.where(np.isnan(fold), -1, fold).astype(int)
    s = np.zeros(len(rows), np.float64); n = np.zeros(len(rows))
    for f in range(4):
        m = AutoModelForSequenceClassification.from_pretrained(os.path.join(WORK9, "ce4", f"ce_e5k4_f{f}"),
                                                               attn_implementation="sdpa").to("cuda")
        sel = np.nonzero((fold == f) | (fold < 0))[0] if split == "train" else np.arange(len(rows))
        s[sel] += predict(m, col, qr[sel], sr[sel]); n[sel] += 1
        del m
    assert (n > 0).all()
    np.save(os.path.join(EA, f"ce_{split}.npy"), (s / n).astype(np.float32))
    print(f"ea ce {split}: {len(rows):,} rows scored", flush=True)


def u(split):
    from ustats import core_of, pair_features
    rows = pd.read_parquet(os.path.join(EA, f"rows_{split}.parquet"), columns=["qid", "s1", "country"])
    st = pd.read_parquet(os.path.join(WORK9, "ustats.parquet"))
    from ustats import STATS
    sp = "test" if split == "test" else "train"
    out = pair_features(core_of(sp, rows.qid.values), core_of(sp, rows.s1.values), rows.country.values,
                        st[["country", "side", "w"] + STATS])
    out.to_parquet(os.path.join(EA, f"u_{split}.parquet"), index=False)
    print(f"ea u {split}: {len(out):,} rows", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "block_fr":
        block_fr()
    else:
        {"block": block, "ce": ce, "u": u}[sys.argv[1]](sys.argv[2])
