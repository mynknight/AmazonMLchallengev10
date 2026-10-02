"""Fast feature experiments: add or replace feature columns on the EXISTING train/val feature
files without re-running blocking (rows are identified by (qid, s1)).

    python augment.py <work> <group>[,<group>...]

Groups
  ph2 : ph_tsort / ph_tset recomputed with the current normalize.phonetic()
  x   : sibling features (features.sibling_features)
  nat : number of native-script words of the query name missing from the transliteration
        dictionary (translit.Transliterator.unknown)
  tok : extra-word log-odds features (features.extra_features).  Training rows get the table
        learned on the OTHER half of the training queries (2-fold by an independent hash);
        validation rows get the table learned on all training rows, saved as extra_words.json.
Writes <work>/feat_{train,val}_aug_<group>.parquet, row-aligned with feat_{train,val}.parquet.
train.py picks them up with ER_AUG=<group>[,...].  The production pipeline computes the same
functions inside `pipeline.py features / predict`; this script only avoids the rebuild while
measuring whether a change helps.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from features import (WORKERS, _cp, extra_features, learn_extra_table, name_vocab_csr, sibling_caches,
                      sibling_features)
from normalize import phonetic
from pipeline import _hash01


def records(work):
    cols = ["entity_id", "nm", "core", "nums"]
    r = pd.concat([pd.read_parquet(os.path.join(work, f"train_s{s}_norm.parquet"), columns=cols)
                   for s in (1, 2, 3)], ignore_index=True)
    return r


def group_tok(r, qi, di, cache, f=None, split=None, work=None):
    if "tok" not in cache:
        csr, _, vocab = name_vocab_csr(r.nm.values, np.array([], dtype=object))
        cache["tok"] = (csr, vocab)
    csr, vocab = cache["tok"]
    if split == "train":
        uq = f.qid.unique()
        fold1 = pd.Series(_hash01(np.char.add(uq.astype(str), "#fold")) < 0.5, index=uq).reindex(f.qid.values).values
        lab = f.label.values
        out = np.zeros((len(f), 6), np.float32)
        for k in (0, 1):
            src, dst = fold1 != bool(k), fold1 == bool(k)
            t = learn_extra_table(csr, csr, vocab, qi[src], di[src], lab[src])
            out[dst] = extra_features(csr, csr, vocab, qi[dst], di[dst], t).values
        cache["tok_table"] = learn_extra_table(csr, csr, vocab, qi, di, lab)
        with open(os.path.join(work, "extra_words.json"), "w", encoding="utf-8") as fh:
            json.dump(cache["tok_table"], fh)
        return pd.DataFrame(out, columns=["w_q_max", "w_q_min", "w_q_n", "w_d_max", "w_d_min", "w_d_n"])
    return extra_features(csr, csr, vocab, qi, di, cache["tok_table"])


def group_ph2(r, qi, di, cache, **_):
    if "ph" not in cache:
        cache["ph"] = np.array([phonetic(c) for c in r.core.values], dtype=object)
    ph = cache["ph"]
    a, b = ph[qi].tolist(), ph[di].tolist()
    return pd.DataFrame({"ph_tsort": _cp(a, b, fuzz.token_sort_ratio, WORKERS),
                         "ph_tset": _cp(a, b, fuzz.token_set_ratio, WORKERS)})


def group_x(r, qi, di, cache, **_):
    if "x" not in cache:
        cache["x"] = sibling_caches(r)
    return sibling_features(qi, di, cache["x"], cache["x"])


def group_nat(r, qi, di, cache, work=None, **_):
    if "nat" not in cache:
        from translit import Transliterator
        tr = Transliterator(os.path.join(work, "translit.json"))
        names = pd.concat([pd.read_parquet(os.path.join(work, f"train_s{s}.parquet"), columns=["business_name"])
                           for s in (1, 2, 3)], ignore_index=True).business_name.values
        cache["nat"] = np.array([tr.unknown(x) for x in names], np.float32)
    return pd.DataFrame({"x_nat_unk_q": cache["nat"][qi]})


GROUPS = {"ph2": group_ph2, "x": group_x, "tok": group_tok, "nat": group_nat}


def main(work, groups):
    t0 = time.time()
    r = records(work)
    pos = pd.Series(np.arange(len(r)), index=r.entity_id.values)
    cache = {}
    for split in ("train", "val"):  # train first: "tok" learns its table there
        f = pd.read_parquet(os.path.join(work, f"feat_{split}.parquet"), columns=["qid", "s1", "label"])
        qi = pos.reindex(f.qid.values).values
        di = pos.reindex(f.s1.values).values
        assert not (np.isnan(qi).any() or np.isnan(di).any()), "unknown ids in feature file"
        qi, di = qi.astype(np.int64), di.astype(np.int64)
        for g in groups:
            out = GROUPS[g](r, qi, di, cache, f=f, split=split, work=work)
            out.to_parquet(os.path.join(work, f"feat_{split}_aug_{g}.parquet"), index=False)
            print(f"  {split} {g}: {len(out):,} rows, cols {list(out.columns)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
