"""Turn blocking output of one partition into a feature frame."""
import numpy as np
import pandas as pd

from features import context_features, build_caches, extra_features, name_vocab_csr, string_features

TEXT_COLS = ["entity_id", "nm", "core", "alias", "ph", "ad", "nums", "state", "nunk"]


def build_features(d, q, qi, di, cn, ca, sel=None, chunk=1_000_000, extra_table=None):
    """Context features use ALL candidates of the partition; string features (the
    expensive part) are computed only for rows in `sel` (boolean mask, default all).
    extra_table (extra_words.json, learned on training pairs by augment.py) adds the
    extra-word features w_*; the training rows get them from augment.py instead (2-fold)."""
    c = pd.DataFrame({"qid": q.entity_id.values[qi], "s1": d.entity_id.values[di],
                      "cos_n": cn, "cos_a": ca})
    context_features(c)
    c["d_name_dup"] = d.core.map(d.core.value_counts()).values[di].astype(np.float32)
    c["d_addr_dup"] = d.ad.map(d.ad.value_counts()).values[di].astype(np.float32)
    if sel is not None:
        c = c[sel].reset_index(drop=True)
        qi, di = qi[sel], di[sel]
    qc, dc = build_caches(q, d)
    res = [string_features(q, d, qi[i:i + chunk], di[i:i + chunk], qc, dc)
           for i in range(0, len(qi), chunk)]
    if res:
        c = pd.concat([c, pd.concat(res, ignore_index=True)], axis=1)
    if extra_table is not None:
        qcsr, dcsr, vocab = name_vocab_csr(q.nm.values, d.nm.values)
        w = extra_features(qcsr, dcsr, vocab, qi, di, extra_table)
        c = pd.concat([c, w], axis=1)
    return c.drop(columns=["_qid_code", "_s1_code"])


def feature_cols(c):
    # state_eq / part are excluded on purpose: France (test-only) never has a parsed
    # state and is always searched country-wide, a combination that in training only
    # occurs for empty-address records -> a spurious signal under distribution shift.
    # x_nat_unk_q drives a hard rule (decide.apply_rules), not the model
    """Model feature columns of a pair-feature frame (ids, label, score and deliberately excluded columns removed)."""
    skip = {"qid", "s1", "label", "country", "prob", "state_eq", "part", "x_nat_unk_q"}
    return [x for x in c.columns if x not in skip and not x.startswith("_")]
