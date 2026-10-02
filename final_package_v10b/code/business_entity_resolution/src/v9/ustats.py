"""Language-agnostic ("unsupervised") word statistics: u_* features.

The v2 extra-word features w_* are learned log-odds of specific words (US / India vocabulary) and are
blank for French, which is why France is scored by a model without them.  Here every word is
described instead by statistics computed on the UNLABELLED candidate pairs of its own country
(test corpus, p1 >= 1e-4), e.g. how often it appears as the only difference between two names at the
same address and house number, whether it replaces another word or is added, and how much more
often it occurs in S2/S3 names than in S1 names.  These statistics mean the same thing in any
language, so a model trained on US / India with them can be applied to France.

    python ustats.py stats            # word statistics per country   -> work/ustats.parquet
    python ustats.py feats <split>    # u_* features for train / val / test rows -> work/u_<split>.parquet
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common9 import CE_WORK, V2, WORK9
from features import EXTRA_SKIP

STATS = ["lcnt", "lcntA", "swap", "solo", "top", "lr", "pbar"]
FEATS = ["u_nq", "u_nd"] + [f"u_{s}_{st}_{a}" for s in ("q", "d") for st in STATS for a in ("min", "max")]


def toks(core):
    return set(w for w in core.split() if w not in EXTRA_SKIP) if isinstance(core, str) else set()


def _extra_chunk(args):
    qc, dc = args
    neq = np.zeros(len(qc), np.int16); ned = np.zeros(len(qc), np.int16); nco = np.zeros(len(qc), np.int16)
    pe, we, se = [], [], []
    for i in range(len(qc)):
        a, b = toks(qc[i]), toks(dc[i])
        eq, ed = a - b, b - a
        neq[i], ned[i], nco[i] = len(eq), len(ed), len(a) - len(eq)
        if len(eq) <= 3 and len(ed) <= 3:
            for w in eq:
                pe.append(i); we.append(w); se.append(0)
            for w in ed:
                pe.append(i); we.append(w); se.append(1)
    return neq, ned, nco, np.array(pe, np.int64), np.array(we, object), np.array(se, np.int8)


def extra_words(qc, dc, procs=48, chunk=200_000):
    """Per pair: #query-extra, #S1-extra, #common tokens; exploded (pair, word, side) of the extras."""
    jobs = [(qc[i:i + chunk], dc[i:i + chunk]) for i in range(0, len(qc), chunk)]
    with Pool(procs) as p:
        res = p.map(_extra_chunk, jobs)
    off = np.cumsum([0] + [len(j[0]) for j in jobs])[:-1]
    neq = np.concatenate([r[0] for r in res]); ned = np.concatenate([r[1] for r in res])
    nco = np.concatenate([r[2] for r in res])
    ex = pd.DataFrame({"pair": np.concatenate([r[3] + o for r, o in zip(res, off)]),
                       "w": np.concatenate([r[4] for r in res]),
                       "side": np.concatenate([r[5] for r in res])})
    return neq, ned, nco, ex


def core_of(split, ids):
    s = pd.concat([pd.read_parquet(os.path.join(V2, f"{split}_s{k}_norm.parquet"), columns=["entity_id", "core"])
                   for k in (1, 2, 3)], ignore_index=True).set_index("entity_id").core
    return s.reindex(ids).values


def stats():
    t0 = time.time()
    import lightgbm as lgb
    c = pd.read_parquet(os.path.join(CE_WORK, "feat_test_p1.parquet"))
    # pbar must mean the same in every country: use the vocabulary-free model's probability
    # everywhere (France's p1 already is; US / India p1 came from the vocabulary model)
    nv = lgb.Booster(model_file=os.path.join(V2, "lgb_v2_novocab.txt"))
    c["p1"] = nv.predict(c[nv.feature_name()].values.astype(np.float32), num_threads=48).astype(np.float32)
    c = c[["qid", "s1", "country", "num_first", "a_tset", "q_cos_sum_rank", "p1"]]
    print("novocab p1 recomputed", flush=True)
    core = pd.concat([pd.read_parquet(os.path.join(V2, f"test_s{k}_norm.parquet"), columns=["entity_id", "core", "country"])
                      for k in (1, 2, 3)], ignore_index=True)
    cs = core.set_index("entity_id").core
    neq, ned, nco, ex = extra_words(cs.reindex(c.qid.values).values, cs.reindex(c.s1.values).values)
    print(f"extras: {len(ex):,} rows ({time.time() - t0:.0f}s)", flush=True)
    near = (nco >= 1) & (neq <= 2) & (ned <= 2)
    A = (c.num_first.values == 1) & (c.a_tset.values >= 90)
    ex = ex[near[ex.pair.values]]
    p = ex.pair.values
    ex = ex.assign(country=c.country.values[p], A=A[p], top=(c.q_cos_sum_rank.values[p] == 1),
                   other=np.where(ex.side.values == 0, ned[p], neq[p]) >= 1,
                   solo=(np.where(ex.side.values == 0, neq[p], ned[p]) == 1) &
                        (np.where(ex.side.values == 0, ned[p], neq[p]) == 0),
                   p1=c.p1.values[p])
    npairs = c.groupby("country").size()
    g = ex.groupby(["country", "side", "w"])
    st = pd.DataFrame({"cnt": g.size(), "cntA": g.A.sum(), "top": g.top.mean()})
    exA = ex[ex.A]
    gA = exA.groupby(["country", "side", "w"])
    st = st.join(pd.DataFrame({"swap": gA.other.mean(), "solo": gA.solo.mean(), "pbar": gA.p1.mean()}))
    st = st.reset_index()
    scale = 1e6 / npairs.reindex(st.country.values).values
    st["lcnt"] = np.log1p(st.cnt.values * scale)
    st["lcntA"] = np.log1p(st.cntA.values * scale)
    # word frequency in S2/S3 names vs S1 names of the same country
    core["src"] = core.entity_id.str.slice(0, 2)
    rec = core.assign(w=core.core.map(lambda x: list(toks(x)))).explode("w").dropna(subset=["w"])
    fr = rec.groupby(["country", "w", "src"]).size().unstack(fill_value=0)
    n = core.groupby(["country", "src"]).size().unstack()
    fr = fr.reset_index()
    f1 = fr["S1"].values / n.reindex(fr.country.values)["S1"].values
    f23 = (fr["S2"].values + fr["S3"].values) / (n.reindex(fr.country.values)["S2"].values +
                                                 n.reindex(fr.country.values)["S3"].values)
    fr["lr"] = np.log((f23 + 1e-5) / (f1 + 1e-5))
    st = st.merge(fr[["country", "w", "lr"]], on=["country", "w"], how="left")
    st.to_parquet(os.path.join(WORK9, "ustats.parquet"), index=False)
    print(f"stats rows {len(st):,} ({time.time() - t0:.0f}s)", flush=True)
    for cn in st.country.unique():
        s = st[(st.country == cn) & (st.side == 0)].sort_values("cntA", ascending=False).head(25)
        print(f"\n{cn}: top query-extra words at the same address\n",
              s[["w", "cnt", "cntA", "swap", "solo", "top", "lr", "pbar"]].round(3).to_string(index=False), flush=True)


def pair_features(qc, dc, country, st):
    """u_* features of pairs given their core names and country."""
    neq, ned, nco, ex = extra_words(qc, dc)
    ex["country"] = np.asarray(country)[ex.pair.values]
    ex = ex.merge(st, on=["country", "side", "w"], how="left")
    ex["lcnt"] = ex.lcnt.fillna(0.0); ex["lcntA"] = ex.lcntA.fillna(0.0)
    out = pd.DataFrame({"u_nq": neq.astype(np.float32), "u_nd": ned.astype(np.float32)})
    for side, s in ((0, "q"), (1, "d")):
        e = ex[ex.side == side]
        g = e.groupby("pair")[STATS]
        for a, agg in (("min", g.min()), ("max", g.max())):
            for stn in STATS:
                col = np.full(len(out), np.nan, np.float32)
                col[agg.index.values] = agg[stn].values
                out[f"u_{s}_{stn}_{a}"] = col
    return out[FEATS]


def feats(split):
    t0 = time.time()
    st = pd.read_parquet(os.path.join(WORK9, "ustats.parquet"))[["country", "side", "w"] + STATS]
    if split == "test":
        pr = pd.read_parquet(os.path.join(CE_WORK, "feat_test_p1.parquet"), columns=["qid", "s1", "country"])
        qc, dc = core_of("test", pr.qid.values), core_of("test", pr.s1.values)
    else:
        pr = pd.read_parquet(os.path.join(CE_WORK, f"pairs_{split}.parquet"), columns=["qid", "s1", "country"])
        qc, dc = core_of("train", pr.qid.values), core_of("train", pr.s1.values)
    out = pair_features(qc, dc, pr.country.values, st)
    out.to_parquet(os.path.join(WORK9, f"u_{split}.parquet"), index=False)
    print(f"u_{split}: {len(out):,} rows, {out.shape[1]} features ({time.time() - t0:.0f}s)", flush=True)
    print(out.describe().T.round(3).to_string(), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "stats":
        stats()
    else:
        feats(sys.argv[2])
