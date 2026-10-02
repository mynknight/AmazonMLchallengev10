"""Candidate generation (blocking).

Every Source 2/3 record (query) retrieves Source 1 records (documents) that share
rare evidence with it in two TF-IDF spaces:

  * name space    : character 3-grams of the normalised core name
  * address space : normalised address tokens + house/plot numbers

The search is partitioned by (country, state) when the query's state is known and
falls back to a country-wide search otherwise.  A numba kernel accumulates the
sparse dot products over inverted lists (skipping extremely frequent features),
then keeps the union of the top-k by name, top-k by address and top-k by the
summed score.  Output: arrays (query_row, doc_row, name_cos, addr_cos).
"""
import numpy as np
import scipy.sparse as sp
from numba import njit, prange, get_num_threads
from sklearn.feature_extraction.text import TfidfVectorizer


def addr_tokens(ad, nums):
    return ad.split() + ["#" + n for n in nums.split()]


def name_text(core, nm):
    return core if core else nm


def fit_spaces(d_name, d_addr):
    vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32,
                         min_df=1)
    va = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, dtype=np.float32,
                         lowercase=False, min_df=1)
    Dn = vn.fit_transform(d_name)
    Da = va.fit_transform(d_addr)
    return vn, va, Dn.tocsc(), Da.tocsc()


@njit(cache=True)
def _insert(ids, sc, k, d, s):
    # ids/sc sorted descending; insert if better than the last
    if s <= sc[k - 1]:
        return
    j = k - 1
    while j > 0 and sc[j - 1] < s:
        sc[j] = sc[j - 1]
        ids[j] = ids[j - 1]
        j -= 1
    sc[j] = s
    ids[j] = d


@njit(parallel=True, cache=True)
def _search(qn_ptr, qn_idx, qn_val, qa_ptr, qa_idx, qa_val,
            dn_ptr, dn_idx, dn_val, da_ptr, da_idx, da_val,
            n_docs, kn, ka, kc, max_df, min_score, nchunks):
    nq = len(qn_ptr) - 1
    K = kn + ka + kc
    out_d = np.full((nq, K), -1, np.int32)
    out_n = np.zeros((nq, K), np.float32)
    out_a = np.zeros((nq, K), np.float32)
    step = (nq + nchunks - 1) // nchunks
    for c in prange(nchunks):
        acc_n = np.zeros(n_docs, np.float32)
        acc_a = np.zeros(n_docs, np.float32)
        seen = np.zeros(n_docs, np.bool_)
        touched = np.empty(n_docs, np.int32)
        bn_i = np.empty(kn, np.int32); bn_s = np.empty(kn, np.float32)
        ba_i = np.empty(ka, np.int32); ba_s = np.empty(ka, np.float32)
        bc_i = np.empty(kc, np.int32); bc_s = np.empty(kc, np.float32)
        for q in range(c * step, min(nq, (c + 1) * step)):
            nt = 0
            for p in range(qn_ptr[q], qn_ptr[q + 1]):
                f = qn_idx[p]
                a, b = dn_ptr[f], dn_ptr[f + 1]
                if b - a > max_df:
                    continue
                w = qn_val[p]
                for r in range(a, b):
                    d = dn_idx[r]
                    if not seen[d]:
                        seen[d] = True
                        touched[nt] = d
                        nt += 1
                    acc_n[d] += w * dn_val[r]
            for p in range(qa_ptr[q], qa_ptr[q + 1]):
                f = qa_idx[p]
                a, b = da_ptr[f], da_ptr[f + 1]
                if b - a > max_df:
                    continue
                w = qa_val[p]
                for r in range(a, b):
                    d = da_idx[r]
                    if not seen[d]:
                        seen[d] = True
                        touched[nt] = d
                        nt += 1
                    acc_a[d] += w * da_val[r]
            bn_i[:] = -1; bn_s[:] = -1.0
            ba_i[:] = -1; ba_s[:] = -1.0
            bc_i[:] = -1; bc_s[:] = -1.0
            for t in range(nt):
                d = touched[t]
                sn = acc_n[d]; sa = acc_a[d]
                if sn + sa >= min_score:
                    _insert(bn_i, bn_s, kn, d, sn)
                    _insert(ba_i, ba_s, ka, d, sa)
                    _insert(bc_i, bc_s, kc, d, sn + sa)
            # write union (dedup) of the three lists
            m = 0
            for lst in range(3):
                if lst == 0:
                    ids = bn_i
                elif lst == 1:
                    ids = ba_i
                else:
                    ids = bc_i
                for j in range(len(ids)):
                    d = ids[j]
                    if d < 0:
                        continue
                    dup = False
                    for z in range(m):
                        if out_d[q, z] == d:
                            dup = True
                            break
                    if not dup:
                        out_d[q, m] = d
                        out_n[q, m] = acc_n[d]
                        out_a[q, m] = acc_a[d]
                        m += 1
            for t in range(nt):
                d = touched[t]
                acc_n[d] = 0.0; acc_a[d] = 0.0; seen[d] = False
    return out_d, out_n, out_a


def search(Dn, Da, Qn, Qa, kn=5, ka=5, kc=10, max_df=20000, min_score=0.15):
    """Dn/Da: CSC doc matrices, Qn/Qa: CSR query matrices (same vocabularies)."""
    Qn = sp.csr_matrix(Qn); Qa = sp.csr_matrix(Qa)
    nchunks = max(1, min(Qn.shape[0], get_num_threads() * 8))
    out_d, out_n, out_a = _search(
        Qn.indptr, Qn.indices, Qn.data, Qa.indptr, Qa.indices, Qa.data,
        Dn.indptr, Dn.indices, Dn.data, Da.indptr, Da.indices, Da.data,
        Dn.shape[0], kn, ka, kc, max_df, min_score, nchunks)
    q, j = np.nonzero(out_d >= 0)
    return q.astype(np.int32), out_d[q, j], out_n[q, j], out_a[q, j]


def block_partition(d, qs, **kw):
    """d: normalised S1 frame of one partition; qs: normalised query frame.

    Returns DataFrame-like dict of candidate pairs (query_row, doc_row, cos_n, cos_a)
    with rows indexing into d / qs.
    """
    d_name = [name_text(c, n) for c, n in zip(d.core.values, d.nm.values)]
    d_addr = [addr_tokens(a, n) for a, n in zip(d.ad.values, d.nums.values)]
    vn, va, Dn, Da = fit_spaces(d_name, d_addr)
    q_name = [name_text(c, n) for c, n in zip(qs.core.values, qs.nm.values)]
    q_addr = [addr_tokens(a, n) for a, n in zip(qs.ad.values, qs.nums.values)]
    Qn = vn.transform(q_name)
    Qa = va.transform(q_addr)
    return search(Dn, Da, Qn, Qa, **kw)
