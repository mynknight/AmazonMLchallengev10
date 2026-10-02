#!/usr/bin/env bash
# End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
#
#   bash run_all.sh [DATA_DIR] [WORK_DIR] [OUT_DIR]
#
# Defaults assume this layout (same as the original Windows machine):
#   <root>/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
#   <root>/finalcode/                              (this folder)
#   <root>/work_v2/                                 (intermediate files, created)
#   <root>/output/                                 (submission files, created)
#
# Every step skips work whose output already exists.  After changing normalize.py /
# translit.py delete WORK/*_norm.parquet and WORK/*_part/; after changing features or
# blocking delete WORK/feat_*.parquet (and retrain).
#
# Env knobs: TRAIN_RATE (share of non-validation train queries used for training, default 0.1;
# raise to 0.3+ on a big-RAM machine), MODEL (model name, default lgb_v1), ER_PROCS
# (normalisation worker processes), ER_THREADS (threads for Numba / rapidfuzz / LightGBM,
# default all cores; cap it on a shared machine), ER_BLOCK_KW (blocking settings, e.g.
# "max_df=20000,kn=5,ka=5,kc=10"; default in pipeline.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$(realpath -m "${1:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}")"
WORK="$(realpath -m "${2:-$ROOT/work_v2}")"
OUT="$(realpath -m "${3:-$ROOT/output}")"
TRAIN_RATE="${TRAIN_RATE:-0.3}"
MODEL="${MODEL:-lgb_v2}"
export ER_THREADS="${ER_THREADS:-$(nproc)}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$ER_THREADS}"
export ER_BLOCK_KW="${ER_BLOCK_KW:-max_df=20000,kn=5,ka=5,kc=10}"   # final (v2) blocking settings
export PYTHONUNBUFFERED=1
mkdir -p "$WORK" "$OUT"
cd "$HERE/src"

step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

WORDS="w_q_max,w_q_min,w_q_n,w_d_max,w_d_min,w_d_n"   # vocabulary-dependent extra-word features
step "1/9 TSV -> parquet";            python io_utils.py "$DATA" "$WORK"
step "2/9 learn transliteration";      [ -f "$WORK/translit.json" ] || python translit.py "$WORK" "$WORK/translit.json"
step "3/9 normalise records";          python prep.py "$WORK" train,test
step "4/9 train/val features";         [ -f "$WORK/feat_val.parquet" ] || python pipeline.py features "$WORK" "$TRAIN_RATE"
step "5/9 extra-word table";           [ -f "$WORK/extra_words.json" ] || python augment.py "$WORK" tok
step "6/9 train LightGBM + tune";      [ -f "$WORK/$MODEL.txt" ] || ER_AUG=tok python train.py "$WORK" "$MODEL"
step "7/9 model for unseen countries"  # same features minus the word table (learned on US / India names)
[ -f "$WORK/${MODEL}_novocab.txt" ] || ER_DROP="$WORDS" python train.py "$WORK" "${MODEL}_novocab"
step "8/9 score test candidates";      python pipeline.py predict "$WORK" test "$WORK/$MODEL.txt" "$WORK/${MODEL}_novocab.txt"
step "9/9 write + validate submission"
python make_submission.py "$WORK" "$DATA" "$OUT" "$WORK/${MODEL}_thresholds.json" "$WORK/${MODEL}_novocab_thresholds.json"
python "$DATA/../utils/validate_submission.py" --matching "$OUT/matching_results.tsv" \
       --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test"
