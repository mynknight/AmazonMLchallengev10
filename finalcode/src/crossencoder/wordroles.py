"""Label-free word-role statistics: a language-independent stand-in for the learned word table.

The v2 extra-word features (w_*) look up per-word log-odds learned from labelled US/India pairs, so
they are blank for French.  What they encode is the ROLE of a word the query adds or drops, and the
data generator gives each role a signature that can be measured without labels, per country, on
"clean" pairs (best candidate of a query, same address, same first house number, names sharing a
word and differing by at most 3 words):
  * generator noise words (center / services / partners; fils / groupe / associes ...) are added
    very often and replace an existing word ~80% of the time (the rest are appended);
  * ordinary name words (technology, trading; club, ecole, amicale ...) are common in source-1
    names, and a swap between two of them at the same address is a sibling distractor
    (8-9% true in US/India validation);
  * distractor words (enterprises, ventures) are appended, never swapped.
Per country and split we count, for every name token t: n_add(t) (query-extra in a clean pair),
n_swap(t) (of those, pairs where the source-1 name also has an extra word), n_drop(t)
(source-1-extra) and n_s1(t) (source-1 names containing t).  LightGBM learns what the rates mean
from the US/India labels; France gets them from its own (test) records.

Per-pair features (NaN when the pair has no such word):
  u_q_add      max over query-extra words of log add-rate (per 10k clean pairs)
  u_q_swap     swap share of that word (shrunk to 0.5)
  u_q_s1       log rate of that word in source-1 names (per 10k names)
  u_q_add_min  min add-rate over query-extra words (an unfamiliar word = typo / invented name)
  u_q_s1_min   min source-1 rate over query-extra words
  u_d_s1_max   max source-1 rate over source-1-extra (dropped) words
  u_d_drop_max max drop-rate over dropped words
"""
import os

import numpy as np
import pandas as pd
from numba import njit, prange

from common import V2
from features import EXTRA_SKIP, name_vocab_csr

U_FEATS = ["u_q_add", "u_q_swap", "u_q_s1", "u_q_add_min", "u_q_s1_min", "u_d_s1_max", "u_d_drop_max"]
MAX_EXTRA = 3


def records(split):
    """Normalised names of all records (sources 1-3) of a split with their country."""
    cols = ["entity_id", "nm", "country"]
    return pd.concat([pd.read_parquet(os.path.join(V2, f"{split}_s{s}_norm.parquet"), columns=cols)
                      for s in (1, 2, 3)], ignore_index=True)


class Tokens:
    """Token-id CSR over all records of a split (sorted ids per record) + entity_id -> row."""

    def __init__(self, split):
        """Tokenise the split's names and keep vocabulary, skip mask, source-1 flag and country per record."""
        r = records(split)
        (self.p, self.ids), _, self.vocab = name_vocab_csr(r.nm.values, np.array([], dtype=object))
        self.pos = pd.Index(r.entity_id.values)
        self.skip = np.array([v in EXTRA_SKIP for v in self.vocab], np.bool_)
        self.is_s1 = r.entity_id.str.startswith("S1-").values
        self.country = r.country.values

    def rows(self, ids):
        """Row numbers of the given entity ids (all must exist)."""
        x = self.pos.get_indexer(ids)
        assert (x >= 0).all()
        return x.astype(np.int64)


@njit(cache=True)
def _walk(qp, qids, dp, dids, a, b, skip, eq, ed):
    """Extra (non-skip) tokens of record a vs record b; returns (n_query_extra, n_s1_extra, common)."""
    i, j, a1, b1 = qp[a], dp[b], qp[a + 1], dp[b + 1]
    ne, nd, common = 0, 0, 0
    while i < a1 or j < b1:
        if j >= b1 or (i < a1 and qids[i] < dids[j]):
            t = qids[i]
            if not skip[t] and ne < 64:
                eq[ne] = t
                ne += 1
            i += 1
        elif i >= a1 or dids[j] < qids[i]:
            t = dids[j]
            if not skip[t] and nd < 64:
                ed[nd] = t
                nd += 1
            j += 1
        else:
            if not skip[qids[i]]:
                common += 1
            i += 1
            j += 1
    return ne, nd, common


@njit(cache=True)
def _role_counts(p, ids, qi, di, skip, max_extra, n_add, n_swap, n_drop):
    """Count, per word, how often it is added, swapped or dropped in clean pairs (same address) with few differing words (Numba)."""
    eq = np.empty(64, np.int64)
    ed = np.empty(64, np.int64)
    for k in range(len(qi)):
        ne, nd, common = _walk(p, ids, p, ids, qi[k], di[k], skip, eq, ed)
        if common == 0 or ne + nd == 0 or ne + nd > max_extra:
            continue
        for x in range(ne):
            n_add[eq[x]] += 1
            if nd > 0:
                n_swap[eq[x]] += 1
        for x in range(nd):
            n_drop[ed[x]] += 1


@njit(parallel=True, cache=True)
def _role_feats(p, ids, qi, di, skip, add, swap, s1r, drop):
    """Seven per-pair word-role features from the query-only and S1-only words and the per-word rate tables (Numba, parallel)."""
    n = len(qi)
    out = np.full((n, 7), np.nan, np.float32)
    for k in prange(n):
        eq = np.empty(64, np.int64)
        ed = np.empty(64, np.int64)
        ne, nd, common = _walk(p, ids, p, ids, qi[k], di[k], skip, eq, ed)
        if ne > 0:
            best, amin, smin = -1, 1e9, 1e9
            for x in range(ne):
                t = eq[x]
                if best < 0 or add[t] > add[best]:
                    best = t
                amin = min(amin, add[t])
                smin = min(smin, s1r[t])
            out[k, 0], out[k, 1], out[k, 2] = add[best], swap[best], s1r[best]
            out[k, 3], out[k, 4] = amin, smin
        if nd > 0:
            smax, dmax = -1e9, -1e9
            for x in range(nd):
                t = ed[x]
                smax = max(smax, s1r[t])
                dmax = max(dmax, drop[t])
            out[k, 5], out[k, 6] = smax, dmax
    return out


def role_tables(tok, pairs):
    """Per-country per-token rates from the clean pairs of one split.
    pairs: qid, s1, p1, country, a_tset, num_first, a_empty_q (all candidates of the split)."""
    b = pairs.sort_values(["qid", "p1"], ascending=[True, False]).drop_duplicates("qid")
    b = b[(b.a_empty_q == 0) & (b.a_tset > 0.95) & (b.num_first > 0.5)]
    V = len(tok.vocab)
    tables = {}
    for c, g in b.groupby("country"):
        n_add, n_swap, n_drop = (np.zeros(V, np.float64) for _ in range(3))
        _role_counts(tok.p, tok.ids, tok.rows(g.qid.values), tok.rows(g.s1.values), tok.skip, MAX_EXTRA,
                     n_add, n_swap, n_drop)
        s1rows = np.nonzero(tok.is_s1 & (tok.country == c))[0]
        lens = tok.p[s1rows + 1] - tok.p[s1rows]
        toks = tok.ids[np.repeat(tok.p[s1rows], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))]
        n_s1 = np.bincount(toks, minlength=V).astype(np.float64)
        nc, ns = len(g), len(s1rows)
        tables[c] = dict(
            add=np.log((n_add + 0.5) / nc * 1e4).astype(np.float32),
            swap=((n_swap + 1.0) / (n_add + 2.0)).astype(np.float32),
            s1=np.log((n_s1 + 0.5) / ns * 1e4).astype(np.float32),
            drop=np.log((n_drop + 0.5) / nc * 1e4).astype(np.float32),
            n_clean=nc, n_s1=ns, n_add=n_add, n_swap=n_swap, n_s1_raw=n_s1)
    return tables


def role_features(tok, tables, qid, s1, country):
    """Word-role features of pairs, using each country's own label-free tables."""
    out = np.full((len(qid), 7), np.nan, np.float32)
    country = np.asarray(country)
    for c, t in tables.items():
        m = country == c
        if m.any():
            out[m] = _role_feats(tok.p, tok.ids, tok.rows(np.asarray(qid)[m]), tok.rows(np.asarray(s1)[m]),
                                 tok.skip, t["add"], t["swap"], t["s1"], t["drop"])
    return pd.DataFrame(out, columns=U_FEATS)


def show(tok, tables, words):
    """Table of the role statistics (add count, swap share, add rate, S1 rate) of selected words per country, for logging."""
    idx = {w: i for i, w in enumerate(tok.vocab)}
    rows = []
    for c, t in tables.items():
        for w in words:
            i = idx.get(w)
            if i is not None:
                rows.append((c, w, int(t["n_add"][i]), round(float(t["swap"][i]), 3),
                             round(float(t["add"][i]), 2), round(float(t["s1"][i]), 2)))
    return pd.DataFrame(rows, columns=["country", "word", "n_add", "swap", "add_rate", "s1_rate"])
