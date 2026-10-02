"""Pair features for (query S2/S3 record, candidate S1 record).

Three groups:
  * string similarity of names / addresses (rapidfuzz, phonetic keys, numbers)
  * blocking scores (TF-IDF cosines)
  * context features: how this candidate ranks among the query's other candidates and
    among all queries pointing at the same S1 entity (encodes the fact that every
    S2/S3 record belongs to at most one S1 entity).
"""
import os

import numpy as np
from numba import njit, prange
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import LEGAL

LEGAL_BITS = {t: 1 << i for i, t in enumerate(sorted(LEGAL))}
# rapidfuzz worker threads (-1 = all cores); ER_THREADS caps it on a shared machine
WORKERS = int(os.environ.get("ER_THREADS", -1))


def _cp(a, b, scorer, workers):
    """Pairwise rapidfuzz scores of aligned lists a[i] vs b[i] (multi-threaded)."""
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _csr_tokens(series_list):
    """Tokenise several string Series with a SHARED vocabulary.

    Returns, per series, (indptr, sorted unique token ids per record, first token id)."""
    lens = [len(s) for s in series_list]
    allv = pd.concat(series_list, ignore_index=True)
    toks = allv.str.split()
    ex = toks.explode()
    rec = ex.index.values
    valid = ex.notna().values
    codes = np.full(len(ex), -1, np.int64)
    codes[valid] = pd.factorize(ex[valid])[0]
    rec, codes = rec[valid], codes[valid]
    counts = np.bincount(rec, minlength=len(allv))
    firsts = np.full(len(allv), -1, np.int64)
    if len(rec):
        start = np.r_[True, rec[1:] != rec[:-1]]
        firsts[rec[start]] = codes[start]
    # sort ids within each record and drop duplicates
    order = np.lexsort((codes, rec))
    rec, codes = rec[order], codes[order]
    keep = np.r_[True, (rec[1:] != rec[:-1]) | (codes[1:] != codes[:-1])] if len(rec) else np.zeros(0, bool)
    rec, codes = rec[keep], codes[keep]
    counts = np.bincount(rec, minlength=len(allv))
    ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    out, off = [], 0
    for n in lens:
        p = ptr[off:off + n + 1]
        out.append((p - p[0], codes[p[0]:p[-1]].astype(np.int32), firsts[off:off + n]))
        off += n
    return out


@njit(parallel=True, cache=True)
def _overlap(qp, qids, dp, dids, qi, di):
    """Token overlap per pair from sorted token-id CSR lists: share of query tokens matched, share of S1 tokens matched, Jaccard (-1 if either side is empty)."""
    n = len(qi)
    out = np.empty((n, 3), np.float32)
    for k in prange(n):
        a0, a1 = qp[qi[k]], qp[qi[k] + 1]
        b0, b1 = dp[di[k]], dp[di[k] + 1]
        la, lb = a1 - a0, b1 - b0
        if la == 0 or lb == 0:
            out[k, 0] = -1.0; out[k, 1] = -1.0; out[k, 2] = -1.0
            continue
        i, j, m = a0, b0, 0
        while i < a1 and j < b1:
            x, y = qids[i], dids[j]
            if x == y:
                m += 1; i += 1; j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        out[k, 0] = m / la
        out[k, 1] = m / lb
        out[k, 2] = m / (la + lb - m)
    return out


@njit(cache=True)
def _popcount(x):
    """Number of set bits in an integer (Numba)."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@njit(parallel=True, cache=True)
def _legal_jac(qm, dm, qi, di):
    """Jaccard of the legal-form bit masks of each pair (-1 if either name has no legal form)."""
    n = len(qi)
    out = np.empty(n, np.float32)
    for k in prange(n):
        a, b = qm[qi[k]], dm[di[k]]
        if a == 0 or b == 0:
            out[k] = -1.0
        else:
            out[k] = _popcount(a & b) / _popcount(a | b)
    return out


def _legal_mask(nm):
    """Bit mask of the legal-form tokens (LLC, Pvt, SARL, ...) in a normalised name."""
    m = 0
    for t in nm.split():
        m |= LEGAL_BITS.get(t, 0)
    return m


def build_caches(q, d):
    """Per-record derived arrays for queries and docs (shared token vocabularies)."""
    (qc_p, qc_i, _), (dc_p, dc_i, _) = _csr_tokens([q.core, d.core])
    (qa_p, qa_i, _), (da_p, da_i, _) = _csr_tokens([q.ad, d.ad])
    (qn_p, qn_i, qn_f), (dn_p, dn_i, dn_f) = _csr_tokens([q.nums, d.nums])
    def longest(s):
        """Longest token of each string (empty string for empty input)."""
        return np.array([max(x.split(), key=len) if x else "" for x in s.values], dtype=object)
    def one(df, cp, ci, ap, ai, np_, ni, nf):
        """Per-record caches for one side of the pairs: token CSRs, numbers, legal mask, spaceless core, token count."""
        return {"core": (cp, ci), "ad": (ap, ai), "nums": (np_, ni), "num_first": nf,
                "num_long": longest(df.nums),
                "legal": np.array([_legal_mask(x) for x in df.nm.values], np.int64),
                "nospace": df.core.str.replace(" ", "", regex=False).values,
                "ntok": (df.core.str.count(" ") + (df.core.str.len() > 0)).values.astype(np.float32)}
    qc, dc = one(q, qc_p, qc_i, qa_p, qa_i, qn_p, qn_i, qn_f), one(d, dc_p, dc_i, da_p, da_i, dn_p, dn_i, dn_f)
    qc["sib"], dc["sib"] = sibling_caches(q), sibling_caches(d)
    return qc, dc


def string_features(q, d, qi, di, qc, dc, workers=WORKERS):
    """Features for pairs (q.iloc[qi], d.iloc[di]); qc/dc from build_caches(q, d)."""
    qi = np.asarray(qi, np.int64); di = np.asarray(di, np.int64)
    n = len(qi)
    out = {}
    def col(df, name, idx):
        """Values of column name for the given row indices, as a list."""
        return df[name].values[idx].tolist()
    a, b = col(q, "core", qi), col(d, "core", di)
    out["n_ratio"] = _cp(a, b, fuzz.ratio, workers)
    out["n_tsort"] = _cp(a, b, fuzz.token_sort_ratio, workers)
    out["n_tset"] = _cp(a, b, fuzz.token_set_ratio, workers)
    out["n_partial"] = _cp(a, b, fuzz.partial_ratio, workers)
    out["n_jw"] = _cp(a, b, JaroWinkler.similarity, workers)
    out["n_exact"] = (q.core.values[qi] == d.core.values[di]).astype(np.float32)
    out["n_len_q"] = qc["ntok"][qi]
    out["n_len_d"] = dc["ntok"][di]
    out["n_nospace"] = _cp(qc["nospace"][qi].tolist(), dc["nospace"][di].tolist(), fuzz.ratio, workers)
    out["n_full_tset"] = _cp(col(q, "nm", qi), col(d, "nm", di), fuzz.token_set_ratio, workers)
    pa_, pb_ = col(q, "ph", qi), col(d, "ph", di)
    out["ph_tsort"] = _cp(pa_, pb_, fuzz.token_sort_ratio, workers)
    out["ph_tset"] = _cp(pa_, pb_, fuzz.token_set_ratio, workers)
    al = q.alias.values[qi]
    has_al = al != ""
    ar = np.full(n, -1.0, np.float32)
    if has_al.any():
        ar[has_al] = _cp(al[has_al].tolist(), d.core.values[di][has_al].tolist(), fuzz.token_sort_ratio, workers)
    out["alias_ratio"] = ar
    sf = _overlap(*qc["core"], *dc["core"], qi, di)
    out["n_cont_q"], out["n_cont_d"], out["n_jac"] = sf.T
    out["legal_q"] = (qc["legal"][qi] != 0).astype(np.float32)
    out["legal_jac"] = _legal_jac(qc["legal"], dc["legal"], qi, di)
    # addresses
    a, b = col(q, "ad", qi), col(d, "ad", di)
    out["a_empty_q"] = (q.ad.values[qi] == "").astype(np.float32)
    out["a_ratio"] = _cp(a, b, fuzz.ratio, workers)
    out["a_tset"] = _cp(a, b, fuzz.token_set_ratio, workers)
    out["a_tsort"] = _cp(a, b, fuzz.token_sort_ratio, workers)
    out["a_partial"] = _cp(a, b, fuzz.partial_ratio, workers)
    sf = _overlap(*qc["ad"], *dc["ad"], qi, di)
    out["a_cont_q"], out["a_cont_d"], out["a_jac"] = sf.T
    nf = _overlap(*qc["nums"], *dc["nums"], qi, di)
    out["num_cont"], out["num_jac"] = nf[:, 0], nf[:, 2]
    f1, f2 = qc["num_first"][qi], dc["num_first"][di]
    out["num_first"] = np.where((f1 < 0) | (f2 < 0), -1.0, f1 == f2).astype(np.float32)
    la, lb = qc["num_long"][qi], dc["num_long"][di]
    nfz = _cp(la.tolist(), lb.tolist(), Levenshtein.normalized_similarity, workers)
    out["num_fuzz"] = np.where((la == "") | (lb == ""), -1.0, nfz).astype(np.float32)
    qs, dsx = q.state.values[qi], d.state.values[di]
    out["state_eq"] = np.where(qs == "", -1.0, (qs == dsx)).astype(np.float32)
    out["src3"] = q.entity_id.str.startswith("S3").values[qi].astype(np.float32)
    # native-script words missing from the transliteration dictionary: a sibling-distractor marker
    out["x_nat_unk_q"] = (q.nunk.values[qi] if "nunk" in q else np.zeros(n)).astype(np.float32)
    out = pd.DataFrame(out)
    return pd.concat([out, sibling_features(qi, di, qc["sib"], dc["sib"], workers)], axis=1)


# --------------------------------------------------------------------- sibling features
# Distractors are often "siblings" of an S1 business: same name, another legal form, a house
# number shifted by a few units (same digit count).  True-pair noise instead looks like typos,
# truncations and a spelling variant of the SAME legal form.  These features separate the two.
LEGAL_CANON = {"private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc",
               "company": "co", "lnc": "inc"}
CANON_BITS = {t: 1 << i for i, t in enumerate(sorted({LEGAL_CANON.get(t, t) for t in LEGAL | set(LEGAL_CANON)}))}
MAXD = 15  # digits kept when a number is parsed into int64


def legal_canon_mask(nm):
    """Bit mask of the canonical legal forms in a name (spelling variants such as limited/ltd share one bit)."""
    m = 0
    for t in nm.split():
        if len(t) > 1 and (t in LEGAL or t in LEGAL_CANON):  # single letters are mostly initials
            m |= CANON_BITS[LEGAL_CANON.get(t, t)]
    return m


def _num_csr(nums):
    """Per record: sorted unique int values of its numbers (CSR), first number and its digit count."""
    vals, ptr = [], [0]
    first = np.full(len(nums), -1, np.int64)
    flen = np.zeros(len(nums), np.int8)
    for i, s in enumerate(nums):
        toks = s.split()
        if toks:
            first[i] = int(toks[0][-MAXD:])
            flen[i] = min(len(toks[0].lstrip("0")) or 1, MAXD)
            vals.extend(sorted({int(t[-MAXD:]) for t in toks}))
        ptr.append(len(vals))
    return np.asarray(ptr, np.int64), np.asarray(vals, np.int64), first, flen


@njit(cache=True)
def _ndig(x):
    """Number of decimal digits of a non-negative integer (Numba)."""
    n = 1
    while x >= 10:
        x //= 10
        n += 1
    return n


@njit(cache=True)
def _digits(x, buf):
    """Write the decimal digits of x into buf (most significant first) and return their count (Numba)."""
    n = _ndig(x)
    for i in range(n - 1, -1, -1):
        buf[i] = x % 10
        x //= 10
    return n


@njit(cache=True)
def _edit1_or_prefix(a, b, ba, bb):
    """(digit edit distance == 1, the digits of one number are a strict prefix of the other's)."""
    na, nb = _digits(a, ba), _digits(b, bb)
    if na > nb:
        ba, bb, na, nb = bb, ba, nb, na
    pref = na < nb
    for i in range(na):
        if ba[i] != bb[i]:
            pref = False
            break
    if na == nb:
        diff = 0
        for i in range(na):
            if ba[i] != bb[i]:
                diff += 1
        return diff == 1, False
    if nb - na != 1:
        return False, pref
    i = 0
    j = 0
    skipped = False
    while i < na and j < nb:
        if ba[i] == bb[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            j += 1
        else:
            return False, pref
    return True, pref


@njit(parallel=True, cache=True)
def _num_unmatched(qp, qv, dp, dv, qi, di):
    """Numbers present on one side only: counts per side, log1p of the smallest |u - v|, whether that
    closest pair has the same digit count, and whether any pair is a 1-digit edit / a truncation."""
    n = len(qi)
    out = np.full((n, 6), -1.0, np.float32)
    for k in prange(n):
        ba = np.empty(20, np.int64)
        bb = np.empty(20, np.int64)
        a0, a1 = qp[qi[k]], qp[qi[k] + 1]
        b0, b1 = dp[di[k]], dp[di[k] + 1]
        if a0 == a1 or b0 == b1:
            continue
        uq = 0
        ud = 0
        best = -1
        bsame = -1.0
        e1 = 0.0
        pf = 0.0
        for i in range(a0, a1):
            hit = False
            for j in range(b0, b1):
                if qv[i] == dv[j]:
                    hit = True
                    break
            if hit:
                continue
            uq += 1
            for j in range(b0, b1):
                found = False
                for i2 in range(a0, a1):
                    if dv[j] == qv[i2]:
                        found = True
                        break
                if found:
                    continue
                dd = abs(qv[i] - dv[j])
                x1, x2 = _edit1_or_prefix(qv[i], dv[j], ba, bb)
                if x1:
                    e1 = 1.0
                if x2:
                    pf = 1.0
                if best < 0 or dd < best:
                    best = dd
                    bsame = 1.0 if _ndig(qv[i]) == _ndig(dv[j]) else 0.0
        for j in range(b0, b1):
            hit = False
            for i in range(a0, a1):
                if dv[j] == qv[i]:
                    hit = True
                    break
            if not hit:
                ud += 1
        out[k, 0] = uq
        out[k, 1] = ud
        if best >= 0:
            out[k, 2] = np.log1p(best)
            out[k, 3] = bsame
            out[k, 4] = e1
            out[k, 5] = pf
    return out


@njit(parallel=True, cache=True)
def _popcount_arr(x):
    """Set-bit count of every element of an integer array (Numba, parallel)."""
    out = np.empty(len(x), np.float32)
    for k in prange(len(x)):
        out[k] = _popcount(x[k])
    return out


def sibling_caches(df):
    """Per-record data for the sibling features: canonical legal mask, house numbers as CSR, first number and its length."""
    p, v, first, flen = _num_csr(df.nums.values)
    return {"lg": np.array([legal_canon_mask(x) for x in df.nm.values], np.int64), "csr": (p, v),
            "first": first, "flen": flen,
            "first_s": np.array([s.split()[0] if s else "" for s in df.nums.values], dtype=object)}


def sibling_features(qi, di, qx, dx, workers=WORKERS):
    """x_* features for pairs; qx/dx from sibling_caches()."""
    qi = np.asarray(qi, np.int64); di = np.asarray(di, np.int64)
    out = {}
    a, b = qx["lg"][qi], dx["lg"][di]
    both = (a != 0) & (b != 0)
    out["x_lg_cjac"] = np.where(both, _popcount_arr(a & b) / np.maximum(_popcount_arr(a | b), 1), -1.0).astype(np.float32)
    out["x_lg_d"] = (b != 0).astype(np.float32)
    f1, f2 = qx["first"][qi], dx["first"][di]
    l1, l2 = qx["flen"][qi].astype(np.int64), dx["flen"][di].astype(np.int64)
    has = (f1 >= 0) & (f2 >= 0)
    out["x_n1_absdiff"] = np.where(has, np.log1p(np.abs(f1 - f2)), -1.0).astype(np.float32)
    out["x_n1_samelen"] = np.where(has, l1 == l2, -1.0).astype(np.float32)
    short, long_ = np.where(l1 <= l2, f1, f2), np.where(l1 <= l2, f2, f1)
    pref = has & (l1 != l2) & (long_ // (10 ** np.clip(np.abs(l1 - l2), 0, 18)) == short)
    out["x_n1_prefix"] = np.where(has, pref, -1.0).astype(np.float32)
    lev = _cp(qx["first_s"][qi].tolist(), dx["first_s"][di].tolist(), Levenshtein.distance, workers)
    out["x_n1_lev"] = np.where(has, lev, -1.0).astype(np.float32)
    u = _num_unmatched(*qx["csr"], *dx["csr"], qi, di)
    (out["x_nu_q"], out["x_nu_d"], out["x_nu_mindiff"], out["x_nu_samelen"], out["x_nu_edit1"],
     out["x_nu_prefix"]) = u.T
    return pd.DataFrame(out)


# ------------------------------------------------------------------ extra-word features
# Which words does the query add to / drop from the S1 name?  Generic words the data generator
# inserts as noise ("center", "services") indicate the same business; a different legal form or
# a location word indicates a sibling.  A per-word log-odds table is learned from labelled
# TRAINING pairs whose names share at least one word (see learn_extra_table).
def name_vocab_csr(q_nm, d_nm):
    """Shared-vocabulary token ids of two name Series: ((qp, qids), (dp, dids), vocab)."""
    allv = pd.concat([pd.Series(q_nm), pd.Series(d_nm)], ignore_index=True)
    ex = allv.str.split().explode()
    valid = ex.notna().values
    codes, vocab = pd.factorize(ex[valid])
    rec = ex.index.values[valid]
    order = np.lexsort((codes, rec))
    rec, codes = rec[order], codes[order]
    keep = np.r_[True, (rec[1:] != rec[:-1]) | (codes[1:] != codes[:-1])] if len(rec) else np.zeros(0, bool)
    rec, codes = rec[keep], codes[keep].astype(np.int32)
    ptr = np.r_[0, np.cumsum(np.bincount(rec, minlength=len(allv)))].astype(np.int64)
    nq = len(q_nm)
    return (ptr[:nq + 1], codes[:ptr[nq]]), (ptr[nq:] - ptr[nq], codes[ptr[nq]:]), np.asarray(vocab, dtype=object)


EXTRA_SKIP = set(LEGAL) | {"and", "the", "smt", "ms", "m/s", "sri", "shri", "shree", "mr", "mrs", "dr",
                           "le", "la", "les", "lnc"}


@njit(cache=True)
def _extra_counts(qp, qids, dp, dids, qi, di, label, skip, max_extra, cnt):
    """cnt[t] += [pos, neg] counts of token t as query-extra (cols 0,1) and S1-extra (cols 2,3),
    over pairs sharing >= 1 non-skip token and differing by <= max_extra non-skip tokens."""
    eq = np.empty(64, np.int32)
    ed = np.empty(64, np.int32)
    for k in range(len(qi)):
        a0, a1 = qp[qi[k]], qp[qi[k] + 1]
        b0, b1 = dp[di[k]], dp[di[k] + 1]
        i, j, common, ne, nd = a0, b0, 0, 0, 0
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
        if common == 0 or ne + nd > max_extra:
            continue
        c = 0 if label[k] else 1
        for x in range(ne):
            cnt[eq[x], c] += 1
        for x in range(nd):
            cnt[ed[x], 2 + c] += 1


@njit(parallel=True, cache=True)
def _extra_feats(qp, qids, dp, dids, qi, di, skip, lo_q, lo_d):
    """Per pair: max / min / count of log-odds over query-extra and S1-extra words (0 if none)."""
    n = len(qi)
    out = np.zeros((n, 6), np.float32)
    for k in prange(n):
        a0, a1 = qp[qi[k]], qp[qi[k] + 1]
        b0, b1 = dp[di[k]], dp[di[k] + 1]
        i, j = a0, b0
        qmax, qmin, qn, dmax, dmin, dn = -99.0, 99.0, 0, -99.0, 99.0, 0
        while i < a1 or j < b1:
            if j >= b1 or (i < a1 and qids[i] < dids[j]):
                t = qids[i]
                if not skip[t]:
                    v = lo_q[t]
                    qmax = max(qmax, v)
                    qmin = min(qmin, v)
                    qn += 1
                i += 1
            elif i >= a1 or dids[j] < qids[i]:
                t = dids[j]
                if not skip[t]:
                    v = lo_d[t]
                    dmax = max(dmax, v)
                    dmin = min(dmin, v)
                    dn += 1
                j += 1
            else:
                i += 1
                j += 1
        if qn:
            out[k, 0], out[k, 1] = qmax, qmin
        if dn:
            out[k, 3], out[k, 4] = dmax, dmin
        out[k, 2], out[k, 5] = qn, dn
    return out


def learn_extra_table(qcsr, dcsr, vocab, qi, di, label, max_extra=3, alpha=20.0, min_count=5):
    """Log-odds (vs the base rate) of each word being a query-extra / S1-extra word in a true pair."""
    skip = np.array([v in EXTRA_SKIP for v in vocab], np.bool_)
    cnt = np.zeros((len(vocab), 4), np.float64)
    _extra_counts(*qcsr, *dcsr, np.asarray(qi, np.int64), np.asarray(di, np.int64),
                  np.asarray(label, np.int8), skip, max_extra, cnt)
    table = {}
    for side, (p, n) in (("q", (0, 1)), ("d", (2, 3))):
        pos, neg = cnt[:, p], cnt[:, n]
        base = min(max(pos.sum() / max(pos.sum() + neg.sum(), 1.0), 1e-4), 1 - 1e-4)
        lo = np.log((pos + alpha * base) / (neg + alpha * (1 - base))) - np.log(base / (1 - base))
        ok = (pos + neg) >= min_count
        table[side] = {vocab[t]: float(lo[t]) for t in np.nonzero(ok)[0]}
    return table


def extra_features(qcsr, dcsr, vocab, qi, di, table):
    """Extra-word log-odds features w_*: max/min/count of the learned log-odds of words present only in the query (w_q_*) or only in the S1 name (w_d_*)."""
    skip = np.array([v in EXTRA_SKIP for v in vocab], np.bool_)
    lo_q = np.array([table["q"].get(v, 0.0) for v in vocab], np.float32)
    lo_d = np.array([table["d"].get(v, 0.0) for v in vocab], np.float32)
    f = _extra_feats(*qcsr, *dcsr, np.asarray(qi, np.int64), np.asarray(di, np.int64), skip, lo_q, lo_d)
    return pd.DataFrame(f, columns=["w_q_max", "w_q_min", "w_q_n", "w_d_max", "w_d_min", "w_d_n"])


def _group_top2(codes, vals):
    """Per-group best and second-best of vals (groups given by integer codes)."""
    order = np.lexsort((-vals, codes))
    cs, vs = codes[order], vals[order]
    first = np.r_[True, cs[1:] != cs[:-1]]
    ng = codes.max() + 1 if len(codes) else 0
    best = np.full(ng, -1.0, np.float32); sec = np.full(ng, -1.0, np.float32)
    best[cs[first]] = vs[first]
    second_pos = np.nonzero(~first & np.r_[False, first[:-1]])[0]
    sec[cs[second_pos]] = vs[second_pos]
    return best[codes], sec[codes]


def _rank_in_group(codes, vals):
    """Rank (1 = highest) of every value within its group code."""
    order = np.lexsort((-vals, codes))
    cs = codes[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    idx = np.arange(len(cs))
    grp_start = np.maximum.accumulate(np.where(start, idx, 0))
    r = np.empty(len(cs), np.float32)
    r[order] = idx - grp_start + 1
    return r


def rank_gap(c, col, key, prefix):
    """Add <prefix>_rank and <prefix>_gap: a pair's rank within its group (query or S1) and its margin to the best/second-best score."""
    codes = pd.factorize(c[key])[0] if f"_{key}_code" not in c else c[f"_{key}_code"].values
    v = c[col].values.astype(np.float32)
    best, sec = _group_top2(codes, v)
    c[f"{prefix}_rank"] = _rank_in_group(codes, v)
    c[f"{prefix}_gap"] = np.where(v >= best, v - sec, v - best).astype(np.float32)
    return c


def context_features(c):
    """c: candidate frame with qid, s1, cos_n, cos_a.  Adds query/entity context in place."""
    c["cos_sum"] = (c.cos_n + c.cos_a).astype(np.float32)
    c["_qid_code"] = pd.factorize(c.qid)[0]
    c["_s1_code"] = pd.factorize(c.s1)[0]
    c["q_ncand"] = c.groupby("_qid_code", sort=False).s1.transform("size").astype(np.float32)
    for col in ("cos_sum", "cos_n", "cos_a"):
        rank_gap(c, col, "qid", f"q_{col}")
    c["d_nq"] = c.groupby("_s1_code", sort=False).qid.transform("size").astype(np.float32)
    rank_gap(c, "cos_sum", "s1", "d_cos_sum")
    return c


def score_context(c, col, prefix):
    """Rank / gap features for a model score column (second-stage model)."""
    rank_gap(c, col, "qid", f"{prefix}_q")
    rank_gap(c, col, "s1", f"{prefix}_d")
    c[f"{prefix}_dsum"] = c.groupby("_s1_code", sort=False)[col].transform("sum").astype(np.float32)
    return c
