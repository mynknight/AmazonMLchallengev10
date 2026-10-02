"""Label-free 'sibling' fingerprint, train vs test: does the test set have more sibling distractors?

A query is classified against the S1 records with the same (country, state, core name):
  exact    : one of them has the same first house number (or both have no number)
  sibling  : one of them shares most address tokens (jaccard >= .5) but the first number differs
  other    : same-core S1 exists but neither of the above
  none     : no same-core S1
Train is also broken down by label to show what each class means.

    python eda_sibling_rate.py <work_dir> [rate=0.1]
"""
import os
import sys

import numpy as np
import pandas as pd

work = sys.argv[1]
rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1


def norm(split, s, cols=("entity_id", "country", "state", "core", "ad", "nums")):
    return pd.read_parquet(os.path.join(work, f"{split}_s{s}_norm.parquet"), columns=list(cols))


def classify(s1, q):
    key = lambda d: d.country + "|" + d.state + "|" + d.core
    s1 = s1[s1.core != ""].assign(k=key, n1=lambda d: d.nums.str.split().str[0].fillna(""))
    q = q.assign(k=key, n1=lambda d: d.nums.str.split().str[0].fillna(""))
    m = q[["entity_id", "k", "ad", "n1"]].merge(s1[["k", "ad", "n1"]], on="k", suffixes=("", "_1"))
    m["jac"] = [len(set(a.split()) & set(b.split())) / max(1, len(set(a.split()) | set(b.split())))
                for a, b in zip(m.ad, m.ad_1)]
    m["exact"] = m.n1 == m.n1_1
    m["sib"] = (~m.exact) & (m.jac >= 0.5)
    g = m.groupby("entity_id").agg(exact=("exact", "any"), sib=("sib", "any"))
    cls = pd.Series("none", index=q.entity_id.values)
    cls[g.index] = np.where(g.exact, "exact", np.where(g.sib, "sibling", "other"))
    return cls


gt = pd.read_parquet(os.path.join(work, "train_gt.parquet"))
matched = set(gt.matched_entity_ids.str.split(",").explode().dropna()) - {""}
rows = []
for split in ("train", "test"):
    s1 = norm(split, 1)
    q = pd.concat([norm(split, 2), norm(split, 3)], ignore_index=True).sample(frac=rate, random_state=0)
    cls = classify(s1, q)
    q["cls"] = cls.reindex(q.entity_id.values).values
    q["grp"] = split if split == "test" else np.where(q.entity_id.isin(matched), "train:matched", "train:distractor")
    for (c, g), d in list(q.groupby(["country", "grp"])) + [((c, split), d) for c, d in q.groupby("country")]:
        vc = d.cls.value_counts(normalize=True)
        rows.append(dict(country=c, group=g, n=len(d), **{k: vc.get(k, 0.0) for k in ("exact", "sibling", "other", "none")}))
r = pd.DataFrame(rows).drop_duplicates(["country", "group"]).set_index(["country", "group"]).sort_index()
print(r.round(4).to_string())
# implied distractor share of test, solving overall = (1-d)*matched + d*distractor per class
for c in ("India", "US"):
    m_, d_, t_ = (r.loc[(c, g)] for g in ("train:matched", "train:distractor", "test"))
    for k in ("exact", "sibling", "none"):
        if abs(d_[k] - m_[k]) > 1e-3:
            print(f"{c}: implied test distractor share from '{k}' rate: {(t_[k] - m_[k]) / (d_[k] - m_[k]):.3f}")
