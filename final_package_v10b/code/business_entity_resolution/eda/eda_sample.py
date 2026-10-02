"""Exploratory analysis on a random sample of the data (default 10% of Source 1 entities).

    python eda_sample.py <work_dir> [rate=0.1] [seed=0]

Needs the parquet + normalised files written by io_utils.py / prep.py.  Sections:
  1. record counts by country, train vs test (all rows; cheap)
  2. label structure of the sampled S1 entities (matches per entity, singletons)
  3. how similar true pairs are (normalised forms), by country and source
  4. how dangerous the distractors (S2/S3 records matching nothing) are
  5. what separates singleton S1 entities from matched ones
  6. do records of the same S1 look more like each other than like the S1 record?
  7. train vs test shift for US / India, and what France looks like
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from translit import NATIVE_RE  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
work = sys.argv[1]
rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0


def raw(split, s, cols=None):
    return pd.read_parquet(os.path.join(work, f"{split}_s{s}.parquet"), columns=cols)


def norm(split, s, cols=None):
    return pd.read_parquet(os.path.join(work, f"{split}_s{s}_norm.parquet"), columns=cols)


def cp(a, b, scorer=fuzz.token_set_ratio):
    return process.cpdist(list(a), list(b), scorer=scorer, workers=16, dtype=np.float32)


def jac(a, b):
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = set(x.split()), set(y.split())
        out[i] = len(x & y) / len(x | y) if (x or y) else np.nan
    return out


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


# --------------------------------------------------------------------------- 1
section("1. records by country (all rows)")
for split in ("train", "test"):
    cnt = pd.DataFrame({f"S{s}": raw(split, s, ["country"]).country.value_counts() for s in (1, 2, 3)})
    cnt = cnt.fillna(0).astype(int)
    cnt["S2+S3 per S1"] = ((cnt.S2 + cnt.S3) / cnt.S1).round(2)
    cnt.loc["total"] = cnt.sum(numeric_only=True)
    cnt.loc["total", "S2+S3 per S1"] = round((cnt.loc["total", "S2"] + cnt.loc["total", "S3"]) / cnt.loc["total", "S1"], 2)
    print(f"\n[{split}]\n{cnt}")

# --------------------------------------------------------------------------- 2
section(f"2. label structure on a {rate:.0%} sample of train S1 entities (seed {seed})")
gt = pd.read_parquet(os.path.join(work, "train_gt.parquet"))
s1n = norm("train", 1)
s1n = s1n.merge(raw("train", 1, ["entity_id", "business_name", "business_address"]), on="entity_id")
samp = s1n.sample(frac=rate, random_state=seed).reset_index(drop=True)
g = gt.set_index("source1_entity_id").matched_entity_ids.reindex(samp.entity_id).fillna("")
lists = g.str.split(",").map(lambda x: [y for y in x if y])
samp["n_s2"] = lists.map(lambda x: sum(y.startswith("S2") for y in x)).values
samp["n_s3"] = lists.map(lambda x: sum(y.startswith("S3") for y in x)).values
samp["n"] = samp.n_s2 + samp.n_s3
print(f"sampled S1 entities: {len(samp):,}")
print("\nmatches per entity by country:")
print(samp.groupby("country").agg(entities=("n", "size"), singleton=("n", lambda x: (x == 0).mean()),
                                  mean_n=("n", "mean"), mean_s2=("n_s2", "mean"), mean_s3=("n_s3", "mean"),
                                  p90_n=("n", lambda x: x.quantile(.9)), max_n=("n", "max")).round(3))
print("\ndistribution of n (share of entities):")
print(pd.crosstab(samp.n.clip(upper=10), samp.country, normalize="columns").round(3).T)
print("\n(n_s2, n_s3) joint, share of non-singletons with 0 from one source:",
      round(((samp.n > 0) & ((samp.n_s2 == 0) | (samp.n_s3 == 0))).sum() / (samp.n > 0).sum(), 3))

# id leakage check: do matched ids track the S1 id?
ex = pd.DataFrame([(a, b) for a, l in zip(samp.entity_id, lists.values) for b in l], columns=["s1", "q"])
num = lambda s: s.str.slice(3).astype(np.int64)
ex_num = pd.DataFrame({"s1": num(ex.s1), "q": num(ex.q), "src": ex.q.str.slice(0, 2)})
for src, d in ex_num.groupby("src"):
    r = np.corrcoef(d.s1.rank(), d.q.rank())[0, 1]
    print(f"id leakage check ({src}): Spearman(S1 id, matched id) = {r:+.4f}")

# --------------------------------------------------------------------------- 3
section("3. similarity of TRUE pairs (normalised forms)")
qn = pd.concat([norm("train", 2), norm("train", 3)], ignore_index=True)
qraw = pd.concat([raw("train", 2, ["entity_id", "business_name", "business_address"]),
                  raw("train", 3, ["entity_id", "business_name", "business_address"])], ignore_index=True)
qn = qn.merge(qraw, on="entity_id")
del qraw
pr = ex.merge(samp.drop(columns=["n", "n_s2", "n_s3"]), left_on="s1", right_on="entity_id") \
       .merge(qn, left_on="q", right_on="entity_id", suffixes=("_1", "_q"))
pr["src"] = pr.q.str.slice(0, 2)
pr["core_eq"] = pr.core_1 == pr.core_q
pr["core_tset"] = cp(pr.core_1, pr.core_q)
pr["core_ratio"] = cp(pr.core_1, pr.core_q, fuzz.ratio)
pr["ph_eq"] = pr.ph_1 == pr.ph_q
pr["ad_jac"] = jac(pr.ad_1, pr.ad_q)
pr["ad_empty_q"] = pr.ad_q == ""
pr["nums_jac"] = jac(pr.nums_1, pr.nums_q)
pr["state_eq"] = np.where(pr.state_q == "", np.nan, (pr.state_1 == pr.state_q).astype(float))
pr["state_missing_q"] = pr.state_q == ""
pr["native_name_q"] = pr.business_name_q.str.contains(NATIVE_RE)
pr["alias_q"] = pr.alias_q != ""
pr["url_like_q"] = pr.business_name_q.str.contains(r"\.(?:com|net|org|in|co)\b|www\.", case=False)
cols = ["core_eq", "ph_eq", "core_tset", "core_ratio", "ad_jac", "ad_empty_q", "nums_jac", "state_eq",
        "state_missing_q", "native_name_q", "alias_q", "url_like_q"]
print(f"true pairs in sample: {len(pr):,}")
print(pr.groupby(["country_1", "src"])[cols].mean().round(3).T)
print("\ncore-name token-set similarity of true pairs (quantiles):")
print(pr.groupby("country_1").core_tset.quantile([.01, .05, .1, .25, .5]).unstack().round(1))
low = pr[pr.core_tset < 50]
print(f"\ntrue pairs with core token-set < 50: {len(low):,} ({len(low)/len(pr):.2%}); examples:")
for r in low.sample(min(15, len(low)), random_state=1).itertuples():
    print(f"  [{r.country_1}] {r.business_name_1!r:45.45} <- {r.business_name_q!r:45.45} | core {r.core_1!r} vs {r.core_q!r}")

# --------------------------------------------------------------------------- 4
section(f"4. distractors (S2/S3 records matching no S1): {rate:.0%} sample of queries")
matched = set(gt.matched_entity_ids.str.split(",").explode().dropna())
qs = qn.sample(frac=rate, random_state=seed).reset_index(drop=True)
qs["is_distractor"] = ~qs.entity_id.isin(matched)
print(qs.groupby("country").is_distractor.mean().round(4).rename("distractor share"))
# is there an S1 record with the SAME normalised core name (same country+state)?
s1_key = s1n.assign(k=s1n.country + "|" + s1n.state + "|" + s1n.core)
core_cnt = s1_key.k.value_counts()
qs["same_core_s1"] = (qs.country + "|" + qs.state + "|" + qs.core).map(core_cnt).fillna(0)
s1_ad = (s1n.country + "|" + s1n.ad).value_counts()
qs["same_addr_s1"] = (qs.country + "|" + qs.ad).map(s1_ad).fillna(0) * (qs.ad != "")
print("\nshare of queries with >=1 S1 of identical (country, state, core name) / identical address tokens:")
print(qs.groupby(["country", "is_distractor"]).agg(
    same_core=("same_core_s1", lambda x: (x > 0).mean()), same_addr=("same_addr_s1", lambda x: (x > 0).mean()),
    native=("business_name", lambda x: x.str.contains(NATIVE_RE).mean()),
    empty_addr=("ad", lambda x: (x == "").mean()), no_state=("state", lambda x: (x == "").mean())).round(3))
# label-free proxy for the matched share, usable on test: does the query have an S1 with the
# same (country, state, core) or the same address tokens?  Compare train vs test on samples.
print("\nlabel-free proxy, train vs test (sample of queries): share with an identical-core / identical-address S1")
rows = []
for split in ("train", "test"):
    s1x = s1n if split == "train" else norm("test", 1)
    kc = (s1x.country + "|" + s1x.state + "|" + s1x.core).value_counts()
    ka = (s1x.country + "|" + s1x.ad).value_counts()
    qx = qs if split == "train" else pd.concat([norm("test", 2), norm("test", 3)], ignore_index=True) \
        .sample(frac=rate, random_state=seed)
    hc = (qx.country + "|" + qx.state + "|" + qx.core).map(kc).fillna(0) > 0
    ha = ((qx.country + "|" + qx.ad).map(ka).fillna(0) > 0) & (qx.ad != "")
    for c in sorted(qx.country.unique()):
        m = (qx.country == c).values
        rows.append(dict(split=split, country=c, same_core=hc[m].mean(), same_addr=ha[m].mean(),
                         either=(hc | ha)[m].mean()))
print(pd.DataFrame(rows).set_index(["country", "split"]).sort_index().round(4))
print("\ndistractor examples (with the S1 records sharing their core name, if any):")
by_core = s1_key.groupby("k")
for r in qs[qs.is_distractor].sample(12, random_state=2).itertuples():
    k = f"{r.country}|{r.state}|{r.core}"
    hits = by_core.get_group(k).head(2) if k in core_cnt.index else s1_key.iloc[:0]
    print(f"  [{r.country}/{r.state}] {r.business_name!r:40.40} | {r.business_address!r:60.60}")
    for h in hits.itertuples():
        print(f"        S1 same core: {h.business_name!r:40.40} | {h.business_address!r:60.60}")

# --------------------------------------------------------------------------- 5
section("5. singleton vs matched S1 entities (sample)")
samp["single"] = samp.n == 0
samp["core_dup_state"] = (samp.country + "|" + samp.state + "|" + samp.core).map(core_cnt)
samp["ad_empty"] = samp.ad == ""
samp["name_ntok"] = samp.core.str.split().str.len()
samp["addr_ntok"] = samp.ad.str.split().str.len()
samp["no_state"] = samp.state == ""
samp["has_nums"] = samp.nums != ""
print(samp.groupby(["country", "single"])[["core_dup_state", "ad_empty", "name_ntok", "addr_ntok", "no_state",
                                           "has_nums"]].mean().round(3))
print("\nsingleton examples:")
for r in samp[samp.single].sample(8, random_state=3).itertuples():
    print(f"  [{r.country}] {r.business_name!r:45.45} | {r.business_address!r}")

# --------------------------------------------------------------------------- 6
section("6. query-query vs query-S1 similarity within the same S1 cluster")
sub = pr[pr.s1.isin(pr.s1.drop_duplicates().sample(min(20000, pr.s1.nunique()), random_state=4))]
a = sub[["s1", "q", "core_q", "ad_q", "core_1", "ad_1"]]
qq = a.merge(a, on="s1", suffixes=("", "_b"))
qq = qq[qq.q < qq.q_b]
print(f"pairs of records of the same S1: {len(qq):,}")
print(f"  name token-set  q-q median {np.median(cp(qq.core_q, qq.core_q_b)):.1f}  vs  q-S1 median {sub.core_tset.median():.1f}")
print(f"  address jaccard q-q mean   {np.nanmean(jac(qq.ad_q, qq.ad_q_b)):.3f}  vs  q-S1 mean   {np.nanmean(sub.ad_jac):.3f}")

# --------------------------------------------------------------------------- 7
section("7. train vs test shift (US / India) and France")
rows = []
for split in ("train", "test"):
    for s in (1, 2, 3):
        d = norm(split, s, ["country", "ad", "state", "core"])
        rw = raw(split, s, ["business_name"])
        d["native"] = rw.business_name.str.contains(NATIVE_RE).values
        for c, x in d.groupby("country"):
            rows.append(dict(split=split, src=f"S{s}", country=c, n=len(x), empty_addr=(x.ad == "").mean(),
                             no_state=(x.state == "").mean(), native_name=x.native.mean(),
                             name_ntok=x.core.str.split().str.len().mean()))
print(pd.DataFrame(rows).set_index(["country", "src", "split"]).sort_index().round(3))
st = []
for split in ("train", "test"):
    d = norm(split, 1, ["country", "state"])
    st.append(d[d.country != "France"].groupby("country").state.value_counts(normalize=True).rename(split))
st = pd.concat(st, axis=1).fillna(0)
st["diff"] = st.test - st.train
print("\nS1 state mix, biggest train->test changes:")
print(st.reindex(st["diff"].abs().sort_values(ascending=False).index).head(10).round(4))

fr = {s: raw("test", s)[lambda x: x.country == "France"] for s in (1, 2, 3)}
print(f"\nFrance: S1 {len(fr[1]):,}  S2 {len(fr[2]):,}  S3 {len(fr[3]):,}")
for s in (1, 2, 3):
    print(f"\n  France S{s} examples:")
    for r in fr[s].sample(8, random_state=5).itertuples():
        print(f"    {r.business_name!r:45.45} | {r.business_address!r}")
f1 = fr[1]
pc = f1.business_address.str.contains(r"\b\d{5}\b")
last = f1.business_address.str.split(",").str[-1].str.strip()
print(f"\n  France S1: 5-digit postcode in address {pc.mean():.3f}; empty address {(f1.business_address == '').mean():.3f}")
print("  most common last address component (region / city):")
print(last.value_counts().head(15).to_string())
legal = f1.business_name.str.extract(r"\b(SARL|SAS|SASU|SA|EURL|SNC|SCI|SELARL|GIE|SCOP)\b", flags=re.I)[0].str.upper()
print("  legal form frequency in France S1 names:\n", legal.value_counts(dropna=False).head(10).to_string())
