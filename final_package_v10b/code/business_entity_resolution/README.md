# Business Entity Resolution: reproduction guide

The pipeline is a cascade.

1. **Normalise** the records: learned word-by-word transliteration of 9 Indian scripts, legal forms,
   aliases, address abbreviations, and US / India / France region parsing.
2. **Block**: (country, state)-partitioned TF-IDF inverted-index search, about 14 candidates per S2/S3
   record.
3. **Learned filter**: a LightGBM on 64 hand features (the stage-1 model) keeps the candidates with
   p ≥ 0.01. This gives the final candidate set in `candidate_pairs.tsv`: 7.51M pairs, 4.34 per S1 entity
   (US/India empty-address records use the deeper search of stage C).
4. **Final matcher** (a LightGBM over the final candidates):
   - US / India (stage C, `src/v9`): the hand features, a 4-fold fine-tuned multilingual
     **cross-encoder** (`intfloat/multilingual-e5-small`, MIT, 118M parameters) with its query / S1
     context, and **language-independent word statistics** (`u_*`: how each differing word behaves in
     the country's unlabelled pairs, not which word it is). Empty-address records get a deeper
     country-wide candidate search (`ea.py`), whose rows replace theirs in the candidate set.
   - France, which does not occur in training (stage B, `fin_nv_u`): hand features plus **label-free
     word-role statistics**, without the cross-encoder or the English/Indian word table.
5. **France correction** (stage D, `src/v10`): the stage-C cross-encoders are adapted to French by
   self-training on the confident France decisions of step 4 (mixed 1:1 with labelled US/India
   pairs). This France cross-encoder then (a) vetoes the France matches whose names differ by one
   unrelated word (`lycee` ↔ `club`) when it gives them < 0.2 (1,360 matches), and (b) adds the
   best candidate of an unmatched France record when the France model gave it 0.6–0.82 (below the
   0.82 threshold) and the cross-encoder gives ≥ 0.98, unless the names differ by such a word swap
   (4,747 matches). On US/India validation, below-threshold matches confirmed this way are 96–99.6% true.
6. **Assign**: every S2/S3 record goes to at most one S1 entity (its best candidate), kept if its
   probability reaches the country threshold tuned for macro F0.5 on a test-like validation set
   (US 0.88, India 0.93, France 0.82).

Public leaderboard (macro F0.5): stage B alone (v6a) 0.988261; + stage C 0.988651 with the veto;
**+ additions (this submission) 0.988932**.

Only the provided data is used: no external lookups, APIs or data. The one external artefact is
the pretrained open model weights (MIT licence, well below 8B parameters), downloaded from the
Hugging Face hub on first use.

## Setup
```bash
conda create -n ber python=3.14 -y && conda activate ber
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
`requirements.txt` is the environment of stages C and D. Stages A and B were originally run in
Python 3.11 with `requirements_py311.txt`; the stage-C outputs were checked to be identical under both.
The model weights are downloaded from the Hugging Face hub on first use (set `HF_HOME` to choose the
cache, `HF_HUB_OFFLINE=1` once they are cached).
A CUDA GPU is needed for the cross-encoder. It was tested on an RTX PRO 6000 with about 14 GB of
GPU memory in use. On a CPU it works but is very slow.

## Run everything
```bash
ER_THREADS=64 bash run_all.sh <dataset_dir> <work_dir> <output_dir>
```
Outputs are `<output_dir>/matching_results.tsv` and `<output_dir>/candidate_pairs.tsv`, and the
official validator runs at the end. Every step skips work whose output already exists. Approximate
times on a 96-core, 500 GB machine: stage A about 2.5 h, stage B about 2 h, stage C about 4.5 h
(4 × cross-encoder training 35 min + scoring 13 min, deeper retrieval 15 min, LightGBM 30 min),
stage D about 1.2 h (4 × France cross-encoder 9–17 min + scoring, submission 12 min). Stage B covers
tokenisation 10 min, 2 × cross-encoder training 2 × 20 min, scoring 20 min, test features 40 min,
word roles 20 min and final models 3 × 10 min. Peak RAM is about 60 GB.

### Stage A (CPU, `src/`)
| step | command | output |
|---|---|---|
| TSV → parquet | `python io_utils.py DATA WORK` | `WORK/{split}_s{k}.parquet`, `train_gt.parquet` |
| transliteration dictionary | `python translit.py WORK WORK/translit.json` | learned from training pairs only |
| normalisation | `python prep.py WORK train,test` | `WORK/{split}_s{k}_norm.parquet` |
| train/val features | `python pipeline.py features WORK 0.3` | `WORK/feat_train.parquet`, `feat_val.parquet` |
| extra-word table | `python augment.py WORK tok` | `WORK/extra_words.json`, `feat_{train,val}_aug_tok.parquet` |
| stage-1 model | `ER_AUG=tok python train.py WORK lgb_v2` | `WORK/lgb_v2.txt`, `lgb_v2_thresholds.json` |
| stage-1 model without word table | `ER_DROP=w_q_max,... python train.py WORK lgb_v2_novocab` | `WORK/lgb_v2_novocab.txt` |
| stage-1 test scores | `python pipeline.py predict WORK test WORK/lgb_v2.txt WORK/lgb_v2_novocab.txt` | `WORK/test_scores.parquet` |

### Stage B (`src/crossencoder/run_ce_all.sh`; work dir `WORK/ce`)
| step | script | what it does |
|---|---|---|
| 1 | `ce_data.py tok` | tokenise every record once: lower-cased "name \| address", native scripts mapped to Latin with the learned dictionary |
| 2 | `ce_data.py p1train`, `ce_data.py pairs` | stage-1 probabilities of the training rows; filtered pair lists; 2 folds by S1 (country, state) |
| 3 | `ce_model.py train/score` | fine-tune one cross-encoder per fold (BCE, 2 epochs); out-of-fold scores for training rows, fold mean for validation / test |
| 4 | `feat_test.py` | the 64 v2 features of the filtered test candidates |
| 5 | `build_roles.py` (`wordroles.py`) | label-free word-role statistics per country and split, and 7 per-pair features |
| 6 | `train_final.py` | final LightGBM: `fin_e5tr` (US/India: v2 features + CE) and `fin_nv_u` (France: v2 features without the word table + word roles) |
| 7 | `predict_final.py` | score the final candidate set (stage-1 p ≥ 0.01) |
| 8 | `make_submission.py` | best S1 per record, country threshold, both TSVs, official validator |

`CE_FRANCE=v3|ce` switches France back to the stage-1 model without the word table, or to the
cross-encoder model. Both were measured on the leaderboard and are worse (see the methodology document).

### Stage C (`src/v9`; work dir `WORK/v9`, cross-encoders in `WORK/v9/ce4`)
| step | command | what it does |
|---|---|---|
| 1 | `ustats.py stats`, `ustats.py feats {val,train,test}` | per-country statistics of every word that differs between two candidate names (unlabelled pairs only) and 7 `u_*` features per pair |
| 2 | `make_folds4.py` | stage-B training pairs in 4 folds by S1 (country, state) |
| 3 | `ce_model4.py train/score e5k4_f{0..3}` | one cross-encoder per fold (trained on 3 folds, 2 epochs); out-of-fold train scores, 4-fold mean for val / test |
| 4 | `ea.py block {train,test}`, `ea.py ce/u {train,val,test}` | empty-address records: country-wide search with max_df 100000, top 40 (recall 72% → 87.6% US, 68% → 91.4% India); features, CE scores and `u_*` of the new rows |
| 5 | `train9.py k4_u_ea --ce ... --no_w --u --ea` | final LightGBM (93 features) + per-country thresholds tuned test-like (validation 0.99057 vs 0.99005 for stage B) |
| 6 | `predict9.py k4_u_ea ...` | US / India test scores → `WORK/v9/test_usin_k4_u_ea.parquet` |

### Stage D (`src/v10`; work dir `WORK/v10`)
| step | command | what it does |
|---|---|---|
| 1 | `prep_fr.py` | France rows of the stage-B test scores (`fin_nv_u`) |
| 2 | `ce_fr10.py train/score {0..3}` | fine-tune each `e5k4_f*` on 1.2M labelled US/India pairs + 739k France pseudo-labels (best candidate with prob ≥ 0.98 → 1, prob ≤ 0.02 → 0), lr 3e-5, 1 epoch; score France test pairs; US/India val log-loss is unchanged (forgetting check) |
| 3 | `veto_fr.py` | report only: share of France matches each cross-encoder would veto, by pattern |
| 4 | `make_sub10.py OUT --v 0.2 --classes swap --add 0.6:0.98` | veto + additions, then `v9/make_sub9.py` writes both TSVs (US/India from stage C) and runs the official validator |

`src/v10/veto_sim.py` (diagnostic) measures the veto idea on US/India validation.

## Validation
Validation is every S1 entity in US {WA, IA, CT} and India {KL, OD}, a geographic hold-out: training
rows never contain these entities, their queries or their distractors. We report plain macro F0.5 and
a **test-like** macro F0.5, where each distractor query counts 1.9 times because the test set has about
40% distractor records against 26% in training. Thresholds are tuned on the test-like variant.

## Diagnostics (not needed to reproduce the outputs)
- `src/crossencoder/evaluate.py`: quick leave-one-state-out check of a cross-encoder signal.
- `src/crossencoder/lbproxy.py`: label-free expected F0.5, which samples the truth from the model's
  own probabilities. It is calibrated on validation.
- `src/analyze_errors.py`, `src/eval_blocking.py`, `src/stage2.py` (measured, not used), `eda/`.
