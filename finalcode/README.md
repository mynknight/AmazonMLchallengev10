# finalcode: code of the final submission (v6a)

This folder holds only the code needed to reproduce the final submission `submissions/v6a_france_roles`
(public leaderboard 0.988261). It contains no data, models or outputs.
- US/India: cascade of TF-IDF blocking → v2 LightGBM filter → final LightGBM with cross-encoder features.
- France (unseen in training): final LightGBM with label-free word-role features instead of the word table.

## Layout
```
finalcode/
├── run_all.sh          stage A: raw TSVs -> blocking -> v2 features -> v2 LightGBM (+ vocabulary-free model)
├── run_ce_all.sh       stage B: cross-encoder + word roles + final LightGBM -> both TSVs + validator
├── requirements.txt    pinned CPU packages (stage B also needs torch + transformers, see below)
└── src/                stage A modules
    ├── io_utils.py       TSV <-> parquet
    ├── translit.py       learned native-script -> Latin word dictionary (Indian scripts)
    ├── normalize.py      name/address normalisation (legal forms, abbreviations, states, leetspeak)
    ├── prep.py           parallel normalisation of every source file
    ├── blocking.py       two TF-IDF spaces (name char 3-grams, address tokens) + Numba top-k search
    ├── run_blocking.py   per (country, state) partitions + country-wide fallback
    ├── features.py       pair features (string similarity, numbers, legal forms, siblings, extra words)
    ├── pairs.py          builds the feature frame of candidate pairs
    ├── pipeline.py       `features` (train/val rows) and `predict` (score test candidates)
    ├── augment.py        extra-word table (w_* features), learned out-of-fold
    ├── train.py          LightGBM training + threshold tuning (standard and test-like validation)
    ├── decide.py         decision rules (best candidate per query, threshold / expected F0.5, hard rules)
    ├── metrics.py        exact macro F0.5
    ├── make_submission.py stage-A submission writer
    └── crossencoder/     stage B modules
        ├── common.py         paths (overridable by environment variables)
        ├── ce_data.py        tokenisation, stage-1 filter, pair lists with folds by S1 state
        ├── ce_model.py       multilingual-e5-small cross-encoder: train (2 folds) and score
        ├── feat_test.py      v2 features of the filtered test candidates
        ├── wordroles.py      label-free word roles (noise / vocabulary / distractor words per country)
        ├── build_roles.py    builds the word-role features u_* for train/val/test
        ├── train_final.py    final LightGBM (v2 features + CE context and/or word roles)
        ├── evaluate.py       validation helpers used by train_final (stage-2 features, reports)
        ├── predict_final.py  scores the test candidates (US/India model, France model)
        └── make_submission.py writes candidate_pairs.tsv + matching_results.tsv, runs the validator
```

## How to run
Expected layout (defaults; every path can be overridden, see the script headers and `crossencoder/common.py`):
`<root>/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/`, `<root>/finalcode/`.

```bash
conda activate ber311                      # Python 3.11 + requirements.txt
ER_THREADS=64 ER_PROCS=48 bash run_all.sh  # stage A, ~2 h; writes <root>/work_v2
conda activate ber_nn                      # ber311 + torch 2.11 (CUDA) + transformers 5.17
ER_V2_WORK=<root>/work_v2 ER_CE_WORK=<root>/ce_work bash run_ce_all.sh <root>/output   # stage B, GPU
```
Final settings (already the defaults): blocking `max_df=20000,kn=5,ka=5,kc=10`, `TRAIN_RATE=0.3`,
model `lgb_v2`, cross-encoder `intfloat/multilingual-e5-small` on transliterated text (`CE_NAME=e5tr`),
France scored by the word-role model (`CE_FRANCE=roles`), thresholds US 0.82 / India 0.85 / France 0.82.
Every step skips outputs that already exist.

Results, experiments and leaderboard analysis: `../INSIGHTS.md`, `../submissions/LEADERBOARD.md`,
`../crossencoder/NOTES.md`, `../business_entity_resolution/HANDOFF.md`.
