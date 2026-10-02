"""Blocking driver: iterate (country, state) partitions and yield candidate pairs.

`iter_partitions` yields, per partition job, the normalised S1 frame `d`, the
normalised query frame `q` (S2+S3) and the candidate arrays (qi, di, cos_n, cos_a)
indexing rows of q / d.  Queries whose state is unknown or absent from Source 1
are searched against the whole country.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (registers pa.compute)
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from blocking import block_partition

COLS = ["entity_id", "country", "state", "nm", "core", "alias", "ph", "ad", "nums", "nunk"]
DEFAULT = "__HIVE_DEFAULT_PARTITION__"


def partition_split(work, split):
    """Write hive-partitioned copies (country/state) of the normalised files."""
    for s in (1, 2, 3):
        dst = os.path.join(work, f"{split}_s{s}_part")
        if os.path.exists(dst):
            continue
        src = pq.ParquetFile(os.path.join(work, f"{split}_s{s}_norm.parquet"))
        schema = src.schema_arrow

        def batches():
            for b in src.iter_batches(batch_size=500_000):
                i = schema.get_field_index("state")
                col = pa.compute.if_else(pa.compute.equal(b.column(i), ""), "none", b.column(i))
                yield b.set_column(i, "state", col)

        ds.write_dataset(batches(), dst, schema=schema, format="parquet",
                         partitioning=["country", "state"], partitioning_flavor="hive",
                         max_rows_per_group=200_000, existing_data_behavior="overwrite_or_ignore")


def _read(work, split, s, country, state=None, exclude_states=None):
    path = os.path.join(work, f"{split}_s{s}_part")
    d = ds.dataset(path, format="parquet", partitioning="hive")
    f = ds.field("country") == country
    if state is not None:
        f = f & (ds.field("state") == state)
    if exclude_states:
        f = f & ~ds.field("state").isin(list(exclude_states))
    df = d.to_table(filter=f, columns=COLS).to_pandas()
    for c in ("country", "state"):
        df[c] = df[c].astype(str)
    df.loc[df.state == "none", "state"] = ""
    return df


def _states(work, split, s, country):
    path = os.path.join(work, f"{split}_s{s}_part", f"country={country}")
    if not os.path.isdir(path):
        return set()
    return {p.split("=", 1)[1] for p in os.listdir(path) if p.startswith("state=")} - {"none"}


def countries_of(work, split):
    return sorted(p.split("=", 1)[1] for p in os.listdir(os.path.join(work, f"{split}_s1_part"))
                  if p.startswith("country="))


def iter_partitions(work, split, countries=None, states=None, **kw):
    partition_split(work, split)
    for country in countries or countries_of(work, split):
        s1_states = _states(work, split, 1, country)
        q_states = _states(work, split, 2, country) | _states(work, split, 3, country)
        jobs = [(st, 0) for st in sorted(s1_states & q_states)]
        jobs.append((None, 1))  # country-wide fallback
        for st, part in jobs:
            if states is not None and part == 0 and st not in states:
                continue
            t0 = time.time()
            if part == 0:
                d = _read(work, split, 1, country, st)
                q = pd.concat([_read(work, split, s, country, st) for s in (2, 3)], ignore_index=True)
            else:
                d = _read(work, split, 1, country)
                q = pd.concat([_read(work, split, s, country, exclude_states=s1_states) for s in (2, 3)],
                              ignore_index=True)
            if len(d) == 0 or len(q) == 0:
                continue
            qi, di, cn, ca = block_partition(d, q, **kw)
            if len(qi) == 0:
                continue
            info = dict(country=country, state=st or "*", part=part, n_docs=len(d), n_q=len(q),
                        n_pairs=len(qi), secs=time.time() - t0)
            yield info, d, q, qi, di, cn, ca


def run(work, split, countries=None, out_name=None, **kw):
    out_path = os.path.join(work, out_name or f"{split}_cands.parquet")
    writer = None
    for info, d, q, qi, di, cn, ca in iter_partitions(work, split, countries, **kw):
        out = pd.DataFrame({"qid": q.entity_id.values[qi], "s1": d.entity_id.values[di],
                            "cos_n": cn, "cos_a": ca, "part": np.int8(info["part"])})
        tab = pa.Table.from_pandas(out, preserve_index=False)
        writer = writer or pq.ParquetWriter(out_path, tab.schema)
        writer.write_table(tab)
        print("  {country}/{state}: docs={n_docs:,} queries={n_q:,} pairs={n_pairs:,} ({secs:.0f}s)"
              .format(**info), flush=True)
    writer.close()


if __name__ == "__main__":
    work, split = sys.argv[1], sys.argv[2]
    countries = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    run(work, split, countries)
