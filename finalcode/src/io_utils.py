"""I/O helpers: TSV <-> Parquet conversion and submission writing."""
import csv
import os

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path, **kw):
    """Read a challenge TSV exactly as written: all columns as strings, no quoting, no NA conversion."""
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_values=[], encoding="utf-8", **kw)


def tsv_to_parquet(src, dst):
    """Convert one TSV to parquet unless the parquet already exists."""
    if os.path.exists(dst):
        return
    df = read_tsv(src)
    df.to_parquet(dst, index=False)
    print(f"  {os.path.basename(src)} -> {len(df):,} rows")


def convert_all(data_dir, work_dir):
    """Convert all train/test source files and the train ground truth to parquet in work_dir."""
    os.makedirs(work_dir, exist_ok=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            tsv_to_parquet(os.path.join(data_dir, split, f"{split}_source{s}.tsv"),
                           os.path.join(work_dir, f"{split}_s{s}.parquet"))
    tsv_to_parquet(os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                   os.path.join(work_dir, "train_gt.parquet"))


def load(work_dir, split, s, columns=None):
    """Load the parquet copy of <split>_source<s> (optionally only some columns)."""
    return pd.read_parquet(os.path.join(work_dir, f"{split}_s{s}.parquet"), columns=columns)


def write_id_lists(path, s1_ids, lists, header):
    """lists: dict s1_id -> iterable of ids."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = lists.get(s1)
            f.write(s1 + "\t" + (",".join(dict.fromkeys(ids)) if ids else "") + "\n")


if __name__ == "__main__":
    import sys
    convert_all(sys.argv[1], sys.argv[2])
