"""Learn a native-script -> Latin word dictionary from the TRAINING ground truth.

Source 2/3 Indian records often carry the business name (and state names in the
address) in Devanagari / Telugu / Kannada / Tamil / Bengali / Gurmukhi / Malayalam /
Gujarati / Odia script.  The transliteration is word-for-word, so for matched pairs
whose token counts agree we align tokens by position and take a majority vote.
Address components (e.g. the state) are learned by co-occurrence with the Source 1
address components of the matched entity.  Unseen native words fall back to
`unidecode` (a phonetic romanisation).  Only the provided training data is used.
"""
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from unidecode import unidecode

NATIVE_RE = re.compile(r"[ऀ-෿]")
NATIVE_WORD_RE = re.compile(r"\S*[ऀ-෿]\S*")


def has_native(s):
    """True if the string contains any Indian native-script character."""
    return bool(NATIVE_RE.search(s))


def _pairs(work_dir, s):
    """Matched (S1 name, S1 addr, Sx name, Sx addr) rows where Sx has native script."""
    gt = pd.read_parquet(os.path.join(work_dir, "train_gt.parquet"))
    gt = gt.assign(t=gt.matched_entity_ids.str.split(",")).explode("t")
    gt = gt[gt.t.str.startswith(f"S{s}-", na=False)][["source1_entity_id", "t"]]
    src = pd.read_parquet(os.path.join(work_dir, f"train_s{s}.parquet"),
                          columns=["entity_id", "business_name", "business_address", "country"])
    src = src[src.country == "India"]
    mask = src.business_name.str.contains(NATIVE_RE) | src.business_address.str.contains(NATIVE_RE)
    src = src[mask]
    m = gt.merge(src, left_on="t", right_on="entity_id")
    s1 = pd.read_parquet(os.path.join(work_dir, "train_s1.parquet"),
                         columns=["entity_id", "business_name", "business_address"])
    m = m.merge(s1, left_on="source1_entity_id", right_on="entity_id", suffixes=("", "_1"))
    return m[["business_name", "business_address", "business_name_1", "business_address_1"]]


def learn(work_dir, out_path, min_count=2):
    """Learn the native->Latin word dictionary and native state/component names by position alignment of train pairs; keep entries seen >= min_count times."""
    word_votes = defaultdict(Counter)
    comp_votes = defaultdict(Counter)
    comp_freq = Counter()
    for s in (2, 3):
        m = _pairs(work_dir, s)
        for nm, ad, nm1, ad1 in m.itertuples(index=False):
            if has_native(nm):
                a, b = nm.split(), nm1.split()
                if len(a) == len(b):
                    for x, y in zip(a, b):
                        if has_native(x):
                            word_votes[x][y.lower()] += 1
            if has_native(ad):
                comps1 = [c.strip().lower() for c in ad1.split(",") if c.strip()]
                comp_freq.update(comps1)
                for c in ad.split(","):
                    c = c.strip()
                    if has_native(c):
                        for c1 in comps1:
                            comp_votes[c][c1] += 1
    words = {w: c.most_common(1)[0][0] for w, c in word_votes.items()
             if sum(c.values()) >= min_count}
    comps = {}
    for c, votes in comp_votes.items():
        n = sum(1 for _ in votes)  # noqa
        total = max(votes.values())
        # pick the S1 component with the best lift (co-occurrence / global freq)
        best = max(votes.items(), key=lambda kv: (kv[1] / (comp_freq[kv[0]] ** 0.5), kv[1]))
        if best[1] >= min_count:
            comps[c] = best[0]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"words": words, "components": comps}, f, ensure_ascii=False)
    print(f"learned {len(words)} words, {len(comps)} address components")
    return words, comps


class Transliterator:
    """Word-for-word transliteration of Indian native-script names and addresses with the learned dictionary."""
    def __init__(self, path):
        """Load the learned word and address-component dictionaries from JSON."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.words = d["words"]
        self.comps = d["components"]

    def unknown(self, s):
        """Native-script words of s missing from the dictionary.  In the training data these are
        a closed set of business-type words (stores, motors, bakery, ...) that the generator adds
        to create sibling distractors; they never occur in a true pair."""
        if not NATIVE_RE.search(s):
            return 0
        return sum(w not in self.words for w in NATIVE_WORD_RE.findall(s))

    def name(self, s):
        """Transliterate a name word by word (unknown words fall back to unidecode)."""
        if not NATIVE_RE.search(s):
            return s
        return NATIVE_WORD_RE.sub(lambda m: self.words.get(m.group(0)) or unidecode(m.group(0)), s)

    def address(self, s):
        """Transliterate an address component by component (known components such as state names first, then word by word)."""
        if not NATIVE_RE.search(s):
            return s
        out = []
        for c in s.split(","):
            cs = c.strip()
            if NATIVE_RE.search(cs):
                cs = self.comps.get(cs) or NATIVE_WORD_RE.sub(
                    lambda m: self.words.get(m.group(0)) or unidecode(m.group(0)), cs)
            out.append(cs)
        return ", ".join(out)


if __name__ == "__main__":
    import sys
    learn(sys.argv[1], sys.argv[2])
