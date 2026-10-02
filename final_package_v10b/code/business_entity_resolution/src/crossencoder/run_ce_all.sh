#!/usr/bin/env bash
# Cross-encoder stage, end to end, on top of a finished v2 run (business_entity_resolution/run_all.sh):
#   tokenise -> stage-1 filter + pair lists -> 2 cross-encoder folds (GPU) -> v2 features of the test
#   candidates -> final LightGBM (+ France variant) -> test scores -> both TSVs + official validator.
#
#   bash run_ce_all.sh [OUT_DIR]
#
# Env: ER_V2_WORK (v2 run dir), ER_CE_WORK (this stage's work dir), ER_DATA (dataset dir),
#      CE_MODEL (default intfloat/multilingual-e5-small), CE_NAME (run name prefix, default e5tr),
#      CE_LR (default 8e-5), CE_FRANCE=roles|v3|ce (countries unseen in training, i.e. France, scored
#      by the label-free word-role model without the CE (default), by the v2 vocabulary-free model = v3,
#      or by the CE model), ER_THREADS (CPU threads, default 24).
# Every step skips work whose output already exists.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(realpath -m "${1:?usage: run_ce_all.sh OUT_DIR}")"
export ER_THREADS="${ER_THREADS:-24}" NUMBA_NUM_THREADS="${ER_THREADS:-24}" PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
export ER_BLOCK_KW="${ER_BLOCK_KW:-max_df=20000,kn=5,ka=5,kc=10}"   # the v2 blocking settings
MODEL="${CE_MODEL:-intfloat/multilingual-e5-small}"
NAME="${CE_NAME:-e5tr}"
LR="${CE_LR:-8e-5}"
FRANCE="${CE_FRANCE:-roles}"
if [ -d "$HERE/src" ]; then cd "$HERE/src"; else cd "$HERE"; fi
W="$(python -c 'from common import WORK; print(WORK)')"
V2="$(python -c 'from common import V2; print(V2)')"
mkdir -p "$W" "$OUT"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "1/8 tokenise records (native scripts transliterated)"
for s in train test; do python ce_data.py tok $s "$MODEL" tr; done
step "2/8 stage-1 probabilities of the training rows, pair lists"
python ce_data.py p1train
[ -f "$W/pairs_test.parquet" ] || python ce_data.py pairs "$MODEL"
step "3/8 cross-encoders, one per fold of source-1 states (GPU)"
for f in 0 1; do
  [ -f "$W/ce_${NAME}_f$f/ce_meta.json" ] || python ce_model.py train ${NAME}_f$f --fold $f --epochs 2 --text tr --model "$MODEL" --lr "$LR"
  for split in val train test; do
    [ -f "$W/ce_${NAME}_f${f}_$split.npy" ] || python ce_model.py score ${NAME}_f$f $split --min_p1 0.009
  done
done
step "4/8 v2 features of the filtered test candidates"
[ -f "$W/feat_test_p1.parquet" ] || python feat_test.py
step "5/8 label-free word-role features (per country and split, no labels)"
[ -f "$W/u_test.parquet" ] || python build_roles.py
step "6/8 final LightGBM (US/India) and the model for countries unseen in training (France)"
CE="${NAME}_f0+${NAME}_f1"
[ -f "$W/fin_$NAME.txt" ] || python train_final.py fin_$NAME "$CE"
[ -f "$W/fin_${NAME}_nv.txt" ] || python train_final.py fin_${NAME}_nv "$CE" --drop "w_*"
[ -f "$W/fin_nv_u.txt" ] || python train_final.py fin_nv_u --no_ce --drop "w_*" --u
step "7/8 score the test candidates"
if [ "$FRANCE" = "roles" ]; then
  python predict_final.py "$W/test_scores_fin_$NAME.parquet" "$W/fin_$NAME.txt" "$W/fin_nv_u.txt" --ce "$CE" --u
  UNSEEN="$W/fin_nv_u_thresholds.json"
elif [ "$FRANCE" = "ce" ]; then
  python predict_final.py "$W/test_scores_fin_$NAME.parquet" "$W/fin_$NAME.txt" "$W/fin_${NAME}_nv.txt" --ce "$CE"
  UNSEEN="$W/fin_${NAME}_nv_thresholds.json"
else
  python predict_final.py "$W/test_scores_fin_$NAME.parquet" "$W/fin_$NAME.txt" --ce "$CE"
  UNSEEN="$V2/lgb_v2_novocab_thresholds.json"
fi
step "8/8 write + validate the submission"
python make_submission.py "$W/test_scores_fin_$NAME.parquet" "$OUT" "$W/fin_${NAME}_thresholds.json" "$UNSEEN"
