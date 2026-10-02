# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** hardmax
**Team Members:**

- Manav Shah, Indian Institute of Technology (IIT), Bombay
- Mayank Choudhary, Indian Institute of Technology (IIT), Bombay
- Riya Jain, Indira Gandhi Delhi Technical University for Women (IGDTUW), Delhi
- Swati Singh, Indira Gandhi Delhi Technical University for Women (IGDTUW), Delhi

**Submission Date:** 2026-09-28

---

## 1. Executive Summary
We resolve Source 2/3 records to Source 1 entities with a cascade. First, a (country, state)-partitioned
TF-IDF blocking step returns about 14 candidates per record; records with an empty address get a deeper
country-wide search. Second, a LightGBM filter over 64 hand-made string, number, legal-form and context
features keeps the candidates with p ≥ 0.01: **4.34 candidates per S1 entity** (the final
`candidate_pairs.tsv`). Third, a final LightGBM matcher scores these candidates. For US / India it adds
a fine-tuned multilingual **cross-encoder** (`multilingual-e5-small`, MIT, 118M parameters, 4 folds)
that reads both raw records, and **language-independent word statistics**. For France, a country absent
from training, it uses **label-free word-role statistics** measured on France's own records and no
cross-encoder. Fourth, a **France-adapted cross-encoder**, self-trained on those France decisions,
corrects them: it vetoes France matches that swap one ordinary name word, and adds below-threshold
France matches it confirms. Each S2/S3 record is assigned to at most one S1 entity and kept if its
probability passes a country threshold tuned for macro F0.5 on a validation set re-weighted to the
test's distractor share.

Public leaderboard: LightGBM alone 0.9797 → + cross-encoder 0.9847 → + France word roles 0.9883 →
+ 4-fold US/India matcher and France veto 0.98865 → + France additions **0.98893**.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Cluster structure** is identical in US and India: 5.6% singletons, 3.46 matches per entity, at most
  11, and the same distribution of match counts to three decimals. Every S2/S3 record matches at most one
  S1 entity.
- **The test set has more distractors:** 40% of S2/S3 records against 26% in training, from the same
  generator.
- **Hard distractors are siblings of a real entity.** They keep the name and change the legal form, shift
  the house number by a few units with the same digit count, append a distractor word (enterprises,
  ventures, group, holdings, traders), or swap one ordinary name word at the same address.
- **True-match noise** includes typos, reordered words, leetspeak, junk prefixes, aliases (`X DBA Y`),
  invented brand names, truncated house numbers, and **generator noise words** (center / services /
  service / partners) that replace a name word about 80% of the time and are appended otherwise.
- **Same address and same house number, one name word differs** (US/India validation): the pair is
  97–100% true when the query word is a noise word, 92–97% for typos and abbreviations, but only
  **8–9% true when two ordinary name words are swapped** (johnson→burke, technology→trading).
- **Native scripts:** 23% of India S2 names and 13% of S3 names are in one of 9 Indian scripts,
  transliterated word by word. Malayalam and Odia occur only in two states.
- **Empty addresses:** records without an address have no state and are searched country-wide by name.
  The standard search retrieves only 72% (US) / 68% (India) of their true pairs, and they account for
  94% of the true pairs never retrieved.
- **France (test only, 15% of S1)** follows the same generator. We measured the same noise-word signature
  (fils, groupe, associés, développement, france: about 78% replacements) and the same vocabulary-swap
  signature (club↔amicale, école↔comité: 98–99% replacements). Its words are French, so no word table
  learned on US/India covers them.
- **Leaderboard probes** (France rows emptied; uploads that change only France) show France was the main
  loss. It scored about 0.946 against about 0.99 for US/India in our earlier submissions, and about
  0.970 after the word-role model.

### 2.2 Solution Strategy
**Approach Type:** Blocking, then a learned filter, then a LightGBM matcher with fine-tuned
cross-encoder features, then constrained assignment, then a cross-encoder correction for France.
**Core Innovations:**
1. A cross-encoder that learns the noise patterns directly from labelled pairs. It is used as a feature
   next to the hand features, with out-of-fold scores from four state folds.
2. Language-independent **word-role features**, computed without labels per country, that let a model
   trained on US/India recognise noise words and vocabulary swaps in French.
3. A **France-adapted cross-encoder**, trained on the confident France decisions of the word-role model,
   used only where it is reliable: to veto word-swap matches it rejects and to confirm
   below-threshold matches.
4. A **test-like validation** (distractors ×1.9) used for every model choice and threshold.

---

## 3. Candidate Generation (Blocking)
- **Normalisation:**
  - Unicode to ASCII.
  - Learned word-by-word transliteration of 9 Indian scripts, from a dictionary built from matched
    training pairs by position alignment.
  - Legal forms and honorifics removed from a `core` name; aliases split off; URLs and hashtags reduced
    to their stem; leetspeak repaired.
  - Address abbreviations canonicalised.
  - State or region parsed and removed from the address tokens: US and Indian states (including `TG`
    and `Keralam`), French regions and départements. A state name that is also a city (Washington,
    Delaware) loses against a different state component.
- **Partitioning:** by (country, state/region). Records without a parsed state are searched country-wide.
- **TF-IDF search:**
  - Two spaces: character 3-grams of the core name, and address tokens plus `#numbers`.
  - A Numba inverted-index kernel computes sparse dot products and skips features with document
    frequency above 20,000.
  - It keeps the union of the top 5 by name, top 5 by address and top 10 by sum: about 14 candidates per
    record, 138.6M test pairs, 80 per S1 entity.
- **Deeper search for empty-address records (US / India):** they are re-blocked country-wide with the
  document-frequency cut raised to 100,000 and the top 40 kept. Validation recall of their true pairs
  rises from 72% to 87.6% (US) and from 68% to 91.4% (India). Their new candidate rows replace the old
  ones in training, validation and test.
- **Learned filter (last candidate stage):**
  - The stage-1 LightGBM (§4.1) scores every blocked pair, and pairs with p ≥ 0.01 are kept.
  - Validation: 94% of blocked pairs dropped (8.26M → 0.50M) for 183 lost true pairs out of 452k
    (0.04%), with unchanged F0.5 for any cut between 1e-4 and 0.03.
  - Test: **7.51M candidate pairs, 4.34 per S1 entity**. This is `candidate_pairs.tsv`, exactly the pairs
    the final matchers run inference on.
- **Recall:** 99.95–99.99% in-state blocking recall for records whose entity is in the same state.

---

## 4. Matching Model

### 4.1 Stage-1 LightGBM (the learned filter, and the base features)
64 features:
- **Name:** rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on core, full and phonetic
  names; token containment and Jaccard.
- **Address:** similarities, containment, empty flag.
- **Numbers:** containment, Jaccard, first-number equality, fuzzy similarity.
- **Sibling features:** canonical legal-form Jaccard; log difference, digit count, truncation and edit
  distance of the first house number; numbers present on one side only.
- **Extra-word log-odds:** a word table learned on US/India labels.
- **Context:** TF-IDF cosines, rank and gap within the record's and the entity's candidates, ambiguity
  of name and address.

LightGBM: 255 leaves, learning rate 0.05, trained on a 30% hash sample of training queries outside
the validation states.

### 4.2 Cross-encoder
- **Model:** `intfloat/multilingual-e5-small` (MIT licence, 118M parameters) with a one-logit
  classification head.
- **Input:** `<s> query </s></s> source-1 </s>`, where each record is `lower(name) | lower(address)`
  with native-script words mapped to Latin by the learned dictionary. Without this mapping, a fold that
  never saw Malayalam collapsed on it (log-loss 1.98).
- **Training pairs:** the filtered candidates of the training rows, 4.3M pairs, 51% positive; the hard
  negatives are the other candidates and the sibling distractors.
- **Training:** BCE loss, AdamW at lr 8e-5, batch 256, 2 epochs, bf16, length-bucketed batches.
- **Folds by source-1 (country, state):** 2 folds in stage B; **4 folds** in the final US/India matcher,
  each model trained on 3 folds (75% of the pairs, about 35 minutes on one GPU). Training rows get the
  out-of-fold score, validation and test the mean of the fold models.
- **Features added to the final model:** the CE logit, and its rank and gap to the best other candidate
  within the record and within the S1 entity.
- **Alone,** the cross-encoder reaches 0.9838 macro F0.5 on validation, and beats the 64-feature model
  on the hardest pairs (AUC 0.915 vs 0.908).

### 4.3 Word-role features (France) and word statistics (US / India)
The learned word table cannot cover an unseen language. It is blank for French, and a model without it
cannot tell a noise word from a vocabulary swap. We replace it with statistics measured **without
labels** on each country's own "clean" pairs (best candidate, same address and first number, names
sharing a word). For every name word we count:
- how often it is added by the record (add-rate);
- how often that addition replaces an S1 word (swap share);
- how common the word is in S1 names.

Seven per-pair features summarise the added and dropped words. LightGBM learns what the statistics
mean from the US/India labels; France receives statistics from its own records.

On validation these features fully replace the labelled word table:

| no word information | learned word table | label-free word roles |
|---|---|---|
| 0.9834 | 0.9859 | **0.9863** |

(test-like macro F0.5)

On France, the resulting model keeps 94% of noise-word pairs and matches only 5% of the 32k vocabulary
swaps. The previous model, without word information, matched 78% of those swaps.

The final US/India matcher uses the same idea (`u_*` features: how often a differing word is the only
difference at the same address, replaced vs added, over-represented in S2/S3 vs S1) in place of the
vocabulary features.

### 4.4 France-adapted cross-encoder: veto and additions
The US/India cross-encoder misreads French: it would veto 30–55% of the France matches that differ by a
French noise word, which are true. We adapt it to French by **self-training**:
- **Pseudo-labels:** the France word-role model's best candidate of a record with p ≥ 0.98 is a positive
  (600k sampled), a France pair with p ≤ 0.02 a negative (139k). The uncertain band is never trained on.
- **Training:** each of the 4 fold models is fine-tuned for one epoch at lr 3e-5 on these pairs mixed
  with 1.2M labelled US/India pairs. US/India validation log-loss is unchanged (0.0454–0.0465 against
  0.0451–0.0459), so it does not forget.
- The adapted model vetoes only 0.3–0.7% of noise-word matches.

It is then used for two narrow corrections of the France decisions (threshold 0.82):
1. **Veto (1,360 matches):** a France match is dropped when the adapted CE gives it < 0.2 **and** the
   two names differ by exactly one unrelated word (edit similarity < 80, after joining spelled-out legal
   forms such as `s c i` → `sci`), e.g. `lycee` ↔ `club`, `pharmacie` ↔ `primaire`. Other vetoes are
   not applied: inspection showed they are mostly true matches (legal-form spacing, noise words, typos).
   On US/India validation, a CE veto on a model of the same type raises held-out test-like F0.5 from
   0.98627 to 0.98813, and the vetoed pairs are 77% false.
2. **Additions (4,747 matches):** a France record left unmatched is matched to its best candidate when
   the word-role model gave it 0.6–0.82 and the adapted CE gives ≥ 0.98, unless the names differ by a
   word swap. On US/India validation such below-threshold matches are only 64–75% true on their own, but
   **96–99.6% true** when the cross-encoder confirms them, well above the ~78% an added pair needs to
   raise F0.5.

### 4.5 Final models and decision
- **US / India:** LightGBM (93 features) on the 64 features minus the vocabulary features, the 4-fold CE
  features and the `u_*` word statistics, with the deeper empty-address candidates (`k4_u_ea`).
- **France:** LightGBM on the 58 language-independent features plus the 7 word-role features
  (`fin_nv_u`), without the cross-encoder, then the veto and additions of §4.4.
- **Decision:** each S2/S3 record is assigned to its most probable S1 entity and kept if
  p ≥ the country threshold, tuned by exact macro F0.5 on the test-like validation: US 0.88,
  India 0.93, France 0.82. Records containing an unknown native-script word (a closed set of 171
  business-type words that only occur in distractors) are never matched.

---

## 5. Results & Error Analysis
Validation: all S1 entities of US {WA, IA, CT} and India {KL, OD} (132,355 entities). "Test-like"
counts every distractor record 1.9 times.

| model (validation macro F0.5) | standard | test-like | US (tl) | India (tl) |
|---|---|---|---|---|
| stage-1 LightGBM (64 features) | 0.9867 | 0.9858 | 0.9859 | 0.9857 |
| same model on the filtered candidates (control) | 0.9867 | 0.9859 | 0.9861 | 0.9858 |
| cross-encoder alone | 0.9838 | 0.9835 | 0.9820 | 0.9861 |
| features + 2-fold cross-encoder (stage B) | 0.9902 | 0.9900 | 0.9891 | 0.9918 |
| **final US/India: 4-fold CE + `u_*` + deeper empty-address search** | **0.9907** | **0.9906** | 0.9896 | 0.9923 |
| France-type model: no word table, no CE | 0.9847 | 0.9834 | 0.9842 | 0.9821 |
| **France model: word roles, no CE** | 0.9870 | 0.9863 | 0.9862 | 0.9864 |

The deeper empty-address search adds +0.0003 to +0.0005 in each of the five held-out states.

Public leaderboard:

| submission | public LB |
|---|---|
| stage-1 LightGBM, France by the model without word table | 0.979663 |
| + cross-encoder for US / India | 0.984698 |
| + word-role model for France | 0.988261 |
| same, France threshold 0.95 instead of 0.82 (probe) | 0.987375 |
| final US/India matcher + France veto | 0.988651 |
| **+ France additions confirmed by the France cross-encoder (final)** | **0.988932** |

Measured and not kept:

| change | result |
|---|---|
| multilingual-e5-base instead of e5-small | better AUC, identical F0.5 |
| unadapted cross-encoder for France | removes true noise-word matches |
| France veto of every match the adapted CE rejects (4,834) | a third are exact names that look true (legal-form spacing, typos); only word swaps kept |
| dropping all 68k France word-swap matches | on US/India, confident swap matches are 100% true; not done |
| second-stage model on probabilities, query consistency, distractor-weighted training, ensembling correlated models, more training rows, per-segment thresholds, 3-seed averaging | +0.0000 to +0.0007 |

**Error analysis** (final US/India model, validation):
- False-positive pairs fell from 1,067 to 469 with the cross-encoder. True pairs missed below the
  threshold fell from 8,865 to 5,810.
- What remains is at the generator's noise floor: house numbers changed by one digit or truncated,
  which are true in some pairs and sibling distractors in others; and invented brand names at the exact
  same address, again true or distractor. A perfect matcher on our candidates would score 0.9981.
- Most never-retrieved true pairs are empty-address records whose names are ambiguous country-wide.

---

## 6. Conclusion
Four things mattered most:
1. Studying the negatives: distractors are siblings built by the generator, and the test has 1.9× more
   of them.
2. A cross-encoder that learns noise patterns end to end, stacked with the hand features rather than
   replacing them.
3. Transferring what the model knows about words to an unseen language through label-free statistics.
   These describe the role a word plays in the data rather than which word it is.
4. Adapting the cross-encoder to the unseen language by self-training, and using it only where
   validation shows it is reliable: vetoing word swaps and confirming near-threshold matches.

A learned filter in front of the final matcher cut the candidate set from 80 to 4.34 per S1 entity
without losing accuracy.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/run_all.sh` runs everything end to end (see `README.md`).
- **Stage A (`src/`):** `io_utils.py`, `translit.py`, `normalize.py` / `prep.py`, `blocking.py` /
  `run_blocking.py`, `features.py` / `pairs.py`, `pipeline.py`, `augment.py`, `train.py`, `decide.py`,
  `metrics.py`.
- **Stage B (`src/crossencoder/`):** `ce_data.py` (tokenisation, pair lists, folds), `ce_model.py`
  (cross-encoder training and scoring), `feat_test.py`, `wordroles.py` / `build_roles.py` (label-free
  word roles), `train_final.py`, `predict_final.py` (France word-role scores), `make_submission.py`,
  `run_ce_all.sh`.
- **Stage C (`src/v9/`, US/India):** `ustats.py` (`u_*` word statistics), `make_folds4.py`,
  `ce_model4.py` (4-fold cross-encoder), `ea.py` (deeper empty-address search), `train9.py`,
  `predict9.py`, `make_sub9.py` (writes both TSVs and runs the official validator).
- **Stage D (`src/v10/`, France):** `prep_fr.py`, `ce_fr10.py` (France-adapted cross-encoder),
  `veto_fr.py` (report), `make_sub10.py` (veto + additions → final TSVs).
- **Diagnostics:** `evaluate.py`, `lbproxy.py`, `analyze_errors.py`, `eval_blocking.py`, `stage2.py`,
  `v10/veto_sim.py`, `eda/`.

### B. Data use
- Only the provided files are used; there are no external lookups or data.
- The pretrained cross-encoder weights are an open MIT-licensed model of 118M parameters.
- Unlabelled test records are used for document frequencies, candidate context features and the
  word-role / word statistics. No test labels exist or are used.
- **Self-training on test predictions (stage D):** the France-adapted cross-encoder is trained on the
  confident France decisions of our own model (pseudo-labels, §4.4), since France has no labelled data.
  No ground truth is involved.
