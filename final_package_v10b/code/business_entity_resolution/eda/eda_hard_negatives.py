"""How do TRUE pairs differ from HARD distractors (same country/state/core name as an S1)?

Checks three candidate signals seen in the samples: legal-form change, house-number change
(numeric distance vs digit edits) and extra words added to the name.

    python eda_hard_negatives.py <work_dir> [rate=0.1]
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

work = sys.argv[1]
rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
LEGAL_CANON = {"private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "corporation": "corp",
               "corp": "corp", "incorporated": "inc", "inc": "inc", "company": "co", "co": "co"}
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from normalize import LEGAL  # noqa: E402


def norm(split, s, cols=None):
    return pd.read_parquet(os.path.join(work, f"{split}_s{s}_norm.parquet"), columns=cols)


def legal_sig(nm):
    return " ".join(sorted({LEGAL_CANON.get(t, t) for t in nm.split() if t in LEGAL}))


gt = pd.read_parquet(os.path.join(work, "train_gt.parquet"))
ex = gt.assign(q=gt.matched_entity_ids.str.split(",")).explode("q")
ex = ex[ex.q.str.len() > 0]
q2s1 = pd.Series(ex.source1_entity_id.values, index=ex.q.values)
s1 = norm("train", 1)
qn = pd.concat([norm("train", 2), norm("train", 3)], ignore_index=True).sample(frac=rate, random_state=0)
qn["true_s1"] = q2s1.reindex(qn.entity_id.values).values

# true pairs of the sampled queries
tp = qn[qn.true_s1.notna()].merge(s1, left_on="true_s1", right_on="entity_id", suffixes=("", "_1"))
# hard distractors: unmatched queries with an S1 of identical (country, state, core); keep the
# S1 whose address shares the most tokens
key = lambda d: d.country + "|" + d.state + "|" + d.core
s1k = s1.assign(k=key(s1))
hd = qn[qn.true_s1.isna() & (qn.core != "")].assign(k=lambda d: key(d)).merge(s1k, on="k", suffixes=("", "_1"))
hd["ov"] = [len(set(a.split()) & set(b.split())) for a, b in zip(hd.ad, hd.ad_1)]
hd = hd.sort_values("ov", ascending=False).drop_duplicates("entity_id")
# the matching control: true pairs whose core is identical too
tpc = tp[tp.core == tp.core_1]
print(f"true pairs {len(tp):,} (identical core {len(tpc):,}); hard distractor pairs {len(hd):,}")


def describe(d, label):
    out = {}
    ls_q, ls_1 = d.nm.map(legal_sig), d.nm_1.map(legal_sig)
    both = (ls_q != "") & (ls_1 != "")
    out["legal both present"] = both.mean()
    out["legal canon equal | both"] = (ls_q == ls_1)[both].mean()
    out["legal raw-token equal | both"] = (d.nm.map(lambda x: " ".join(sorted(t for t in x.split() if t in LEGAL)))
                                          == d.nm_1.map(lambda x: " ".join(sorted(t for t in x.split() if t in LEGAL))))[both].mean()
    f_q = d.nums.str.split().str[0]
    f_1 = d.nums_1.str.split().str[0]
    has = f_q.notna() & f_1.notna()
    eq = (f_q == f_1) & has
    out["first num both present"] = has.mean()
    out["first num equal | both"] = eq[has].mean()
    diff = has & ~eq
    a = f_q[diff].astype(float).values
    b = f_1[diff].astype(float).values
    lev = np.array([Levenshtein.distance(x, y) for x, y in zip(f_q[diff], f_1[diff])])
    ad = np.abs(a - b)
    out["| differ: abs diff <= 10"] = (ad <= 10).mean() if len(ad) else np.nan
    out["| differ: abs diff <= 100"] = (ad <= 100).mean() if len(ad) else np.nan
    out["| differ: 1 digit edit"] = (lev == 1).mean() if len(ad) else np.nan
    out["| differ: 1 edit & diff>10"] = ((lev == 1) & (ad > 10)).mean() if len(ad) else np.nan
    out["| differ: same length"] = (f_q[diff].str.len().values == f_1[diff].str.len().values).mean() if len(ad) else np.nan
    out["| differ: prefix/suffix (truncation)"] = np.mean([x.startswith(y) or y.startswith(x) for x, y in zip(f_q[diff], f_1[diff])]) if len(ad) else np.nan
    all_nums_eq = d.nums == d.nums_1
    out["all nums identical"] = all_nums_eq.mean()
    out["address tokens identical"] = (d.ad == d.ad_1).mean()
    return pd.Series(out, name=label)


print(pd.concat([describe(tp, "true (all)"), describe(tpc, "true (same core)"), describe(hd, "hard distractor")],
                axis=1).round(3).to_string())


def extra_tokens(d):
    c = Counter()
    for a, b in zip(d.nm, d.nm_1):
        c.update(set(a.split()) - set(b.split()))
    return c


# extra words of the query's FULL name (beyond the S1 full name), true vs any distractor-vs-nearest
ct = extra_tokens(tp)
# for distractors use every unmatched query with a same-state S1 sharing >= half its core tokens: approximated
# here by hard distractors plus their full-name differences
cd = extra_tokens(hd)
nt, nd = len(tp), len(hd)
rows = []
for t in set(ct) | set(cd):
    a, b = ct.get(t, 0), cd.get(t, 0)
    if a + b >= 30:
        rows.append((t, a, b, a / nt, b / nd, np.log((b / nd + 1e-5) / (a / nt + 1e-5))))
r = pd.DataFrame(rows, columns=["token", "n_true", "n_hard", "rate_true", "rate_hard", "log_odds_hard"])
print("\nextra name tokens most typical of HARD DISTRACTORS:")
print(r.sort_values("log_odds_hard", ascending=False).head(30).round(4).to_string(index=False))
print("\nextra name tokens most typical of TRUE pairs:")
print(r.sort_values("log_odds_hard").head(20).round(4).to_string(index=False))
print("\nshare of pairs with >=1 extra token: true %.3f, hard %.3f" %
      (np.mean([bool(set(a.split()) - set(b.split())) for a, b in zip(tp.nm, tp.nm_1)]),
       np.mean([bool(set(a.split()) - set(b.split())) for a, b in zip(hd.nm, hd.nm_1)])))
