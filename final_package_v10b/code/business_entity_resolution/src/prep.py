"""Normalise every source file into <work>/<split>_s<k>_norm.parquet (chunked, multiprocess)."""
import os
import sys
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalize import normalize_frame
from translit import Transliterator

_TR = None


def _init(path):
    global _TR
    _TR = Transliterator(path)


def _work(df):
    return normalize_frame(df, _TR)


def prep(work, split, s, procs=int(os.environ.get("ER_PROCS", min(os.cpu_count() or 4, 16))),
         chunk=100_000):
    dst = os.path.join(work, f"{split}_s{s}_norm.parquet")
    if os.path.exists(dst):
        return
    src = pq.ParquetFile(os.path.join(work, f"{split}_s{s}.parquet"))
    writer = None
    batches = (b.to_pandas() for b in src.iter_batches(batch_size=chunk))
    with Pool(procs, initializer=_init, initargs=(os.path.join(work, "translit.json"),)) as pool:
        for out in pool.imap(_work, batches):
            t = pa.Table.from_pandas(out, preserve_index=False)
            writer = writer or pq.ParquetWriter(dst, t.schema)
            writer.write_table(t)
    writer.close()
    print(f"  normalised {split} s{s}", flush=True)


if __name__ == "__main__":
    work = sys.argv[1]
    for split in sys.argv[2].split(","):
        for s in (1, 2, 3):
            prep(work, split, s)
