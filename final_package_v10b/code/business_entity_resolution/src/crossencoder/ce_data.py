"""Inputs for the cross-encoder: tokenised records and the pair lists it is trained / run on.

    python ce_data.py tok <split> [model]   # every S1/S2/S3 record of the split -> token ids (CSR)
    python ce_data.py p1train               # v2 LightGBM probability of every training row (stage-1 filter)
    python ce_data.py pairs                 # pairs_{train,val,test}.parquet

Record text = lower-cased raw "name | address" (whitespace collapsed); native scripts are kept, the
model has to learn transliteration itself.  Each record is tokenised once (no special tokens,
at most MAXREC tokens); a pair is assembled on the fly as <s> query </s></s> source-1 </s>.

Pairs are the v2 blocking candidates that pass the stage-1 filter (v2 probability >= P1_MIN):
  train: rows of work_v2/feat_train.parquet (30% query sample, training states); the v2 model was
         trained on these rows, so its probability is in-sample and the filter is looser (1e-5)
  val  : rows of work_v2/val_scores_lgb_v2.parquet (validation states, honest probabilities)
  test : rows of work_v2/test_scores.parquet (France scored by the vocabulary-free model)
Each training pair gets a fold (0/1) from the (country, state) of its source-1 entity, so two
cross-encoders trained on one fold each give out-of-fold scores for every training row.
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import P1_MIN, RAW, THREADS, V2, WORK

MAXREC = 72
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
TRAIN_P1_MIN = 1e-5
_TOK = None


def record_text(df, translit=False):
    """lower-cased "name | address"; translit=True first maps native-script words to Latin with the
    dictionary learned from the training pairs (translit.py; unknown words -> unidecode), so the
    model does not depend on having seen each script (Malayalam / Odia occur only in the
    validation states)."""
    name, addr = df.business_name.fillna(""), df.business_address.fillna("")
    if translit:
        from translit import NATIVE_RE, Transliterator
        tr = Transliterator(os.path.join(V2, "translit.json"))
        name, addr = name.copy(), addr.copy()
        m = name.str.contains(NATIVE_RE).values
        name[m] = [tr.name(x) for x in name[m]]
        m = addr.str.contains(NATIVE_RE).values
        addr[m] = [tr.address(x) for x in addr[m]]
    clean = lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return (clean(name) + " | " + clean(addr)).tolist()


def load_records(split):
    """All records of a split in a fixed order: S1, S2, S3 (row number = record index)."""
    parts = [pd.read_parquet(os.path.join(RAW, f"{split}_s{s}.parquet")) for s in (1, 2, 3)]
    return pd.concat(parts, ignore_index=True)


def _tok_init(model):
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(model)


def _tok_chunk(texts):
    enc = _TOK(texts, add_special_tokens=False, truncation=True, max_length=MAXREC)["input_ids"]
    lens = np.fromiter((len(x) for x in enc), np.int32, len(enc))
    flat = np.fromiter(itertools.chain.from_iterable(enc), np.int32, int(lens.sum()))
    return lens, flat


def model_tag(model, text="raw"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    # models sharing a vocabulary (XLM-R: e5, MiniLM, XLM-R) share the file
    return f"v{tok.vocab_size}" + ("" if text == "raw" else f"_{text}")


def tokenize(split, model=DEFAULT_MODEL, text="raw", chunk=50_000):
    dst = os.path.join(WORK, f"tok_{split}_{model_tag(model, text)}.npz")
    if os.path.exists(dst):
        print("exists", dst)
        return
    t0 = time.time()
    rec = load_records(split)
    texts = record_text(rec, translit=(text == "tr"))
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    with Pool(min(THREADS, 32), initializer=_tok_init, initargs=(model,)) as pool:
        res = pool.map(_tok_chunk, chunks)
    lens = np.concatenate([r[0] for r in res])
    ids = np.concatenate([r[1] for r in res])
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    np.savez(dst, ids=ids, off=off, entity_id=rec.entity_id.values.astype(str))
    print(f"tokenised {split}: {len(lens):,} records, {len(ids):,} tokens, mean {lens.mean():.1f}, "
          f"truncated {(lens == MAXREC).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)


def p1_train():
    """In-sample v2 probability for every row of feat_train (aligned by row)."""
    import lightgbm as lgb
    from decide import apply_rules
    dst = os.path.join(WORK, "p1_train.npy")
    if os.path.exists(dst):
        return
    bst = lgb.Booster(model_file=os.path.join(V2, "lgb_v2.txt"))
    feats = bst.feature_name()
    aug = pd.read_parquet(os.path.join(V2, "feat_train_aug_tok.parquet"))
    base = [f for f in feats if f not in aug.columns] + ["x_nat_unk_q"]
    pf = pq.ParquetFile(os.path.join(V2, "feat_train.parquet"))
    out, pos = [], 0
    t0 = time.time()
    for g in range(pf.num_row_groups):
        df = pf.read_row_group(g, columns=base).to_pandas()
        n = len(df)
        for c in aug.columns:
            df[c] = aug[c].values[pos:pos + n]
        out.append(apply_rules(df, bst.predict(df[feats].values.astype(np.float32), num_threads=THREADS))
                   .astype(np.float32))
        pos += n
        if g % 10 == 0:
            print(f"  p1 row group {g}/{pf.num_row_groups} ({time.time() - t0:.0f}s)", flush=True)
    assert pos == len(aug)
    np.save(dst, np.concatenate(out))


def _rows(index, ids):
    r = index.get_indexer(ids)
    assert (r >= 0).all(), "pair id missing from the tokenised records"
    return r.astype(np.int32)


def _folds(tr, s1_state):
    """(country, state) of the source-1 entity -> fold, greedy balance of pair counts per country."""
    key = tr.country.astype(str) + "/" + s1_state
    cnt = key.value_counts()
    fold, load = {}, {}
    for k, n in cnt.items():
        c = k.split("/")[0]
        l0, l1 = load.get((c, 0), 0), load.get((c, 1), 0)
        f = 0 if l0 <= l1 else 1
        fold[k] = f
        load[(c, f)] = load.get((c, f), 0) + n
    return key.map(fold).values.astype(np.int8)


def pairs(model=DEFAULT_MODEL):
    tag = model_tag(model)
    idx = {}
    for split in ("train", "test"):
        z = np.load(os.path.join(WORK, f"tok_{split}_{tag}.npz"))
        idx[split] = pd.Index(z["entity_id"])

    # --- train
    tr = pd.read_parquet(os.path.join(V2, "feat_train.parquet"), columns=["qid", "s1", "label", "country"])
    tr["p1"] = np.load(os.path.join(WORK, "p1_train.npy"))
    tr["row"] = np.arange(len(tr), dtype=np.int64)
    tr = tr[tr.p1 >= TRAIN_P1_MIN].reset_index(drop=True)
    s1n = pd.read_parquet(os.path.join(V2, "train_s1_norm.parquet"), columns=["entity_id", "state"])
    st = s1n.set_index("entity_id").state.reindex(tr.s1.values).fillna("").values
    tr["fold"] = _folds(tr, pd.Series(st).replace("", "none").values)
    tr["q_row"] = _rows(idx["train"], tr.qid.values)
    tr["s_row"] = _rows(idx["train"], tr.s1.values)
    tr.to_parquet(os.path.join(WORK, "pairs_train.parquet"), index=False)
    print(f"train pairs {len(tr):,} pos {tr.label.mean():.3f} | fold sizes "
          f"{np.bincount(tr.fold).tolist()} | queries {tr.qid.nunique():,}", flush=True)

    # --- val
    va = pd.read_parquet(os.path.join(V2, "val_scores_lgb_v2.parquet"))
    va = va.rename(columns={"prob": "p1"})
    va["row"] = np.arange(len(va), dtype=np.int64)
    va = va[va.p1 >= P1_MIN].reset_index(drop=True)
    va["q_row"] = _rows(idx["train"], va.qid.values)
    va["s_row"] = _rows(idx["train"], va.s1.values)
    va.to_parquet(os.path.join(WORK, "pairs_val.parquet"), index=False)
    print(f"val pairs {len(va):,} pos {va.label.mean():.3f}", flush=True)

    # --- test
    from pipeline import decode_ids
    te = pd.read_parquet(os.path.join(V2, "test_scores.parquet"))
    te = te[te.prob >= P1_MIN].reset_index(drop=True).rename(columns={"prob": "p1"})
    te["qid"] = decode_ids(te.q.values)
    te["s1"] = decode_ids(te.s1.values, "S1-")
    te = te.drop(columns=["q"])
    te["q_row"] = _rows(idx["test"], te.qid.values)
    te["s_row"] = _rows(idx["test"], te.s1.values)
    te.to_parquet(os.path.join(WORK, "pairs_test.parquet"), index=False)
    print(f"test pairs {len(te):,} by country {te.country.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "tok":  # tok <split> [model] [raw|tr]
        tokenize(sys.argv[2], *(sys.argv[3:4] or [DEFAULT_MODEL]), *(sys.argv[4:5] or ["raw"]))
    elif cmd == "p1train":
        p1_train()
    elif cmd == "pairs":
        pairs(*(sys.argv[2:3] or [DEFAULT_MODEL]))
