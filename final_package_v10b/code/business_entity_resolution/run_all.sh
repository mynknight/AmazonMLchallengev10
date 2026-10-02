#!/usr/bin/env bash
# End-to-end pipeline: raw TSVs -> OUT_DIR/matching_results.tsv + OUT_DIR/candidate_pairs.tsv
#
#   bash run_all.sh [DATA_DIR] [WORK_DIR] [OUT_DIR]
#
# Stage A (CPU): normalisation, (country, state) TF-IDF blocking, 64 hand features, LightGBM
#   stage-1 models (the learned candidate filter) and stage-1 scores of every test candidate.
# Stage B (GPU): src/crossencoder/run_ce_all.sh - 2-fold cross-encoder, label-free word-role
#   features, final LightGBM; its France scores (fin_nv_u, "v6a") are the France model.
# Stage C (GPU): src/v9 - US/India final matcher: 4-fold cross-encoder, language-independent word
#   statistics (u_*), deeper retrieval for empty-address records, LightGBM k4_u_ea.
# Stage D (GPU): src/v10 - France: cross-encoder adapted to France by self-training on the v6a
#   decisions; veto of v6a word-swap matches it rejects, additions of below-threshold matches it
#   confirms; writes both TSVs and runs the official validator.
# Defaults assume this layout:
#   <root>/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv
#   <root>/business_entity_resolution/   (this folder)
# Every step of stages A and B skips work whose output already exists.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$(realpath -m "${1:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}")"
WORK="$(realpath -m "${2:-$ROOT/work}")"
OUT="$(realpath -m "${3:-$ROOT/output}")"
export ER_THREADS="${ER_THREADS:-$(nproc)}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$ER_THREADS}"
export ER_PROCS="${ER_PROCS:-16}"
export ER_BLOCK_KW="${ER_BLOCK_KW:-max_df=20000,kn=5,ka=5,kc=10}"
export ER_ROUNDS="${ER_ROUNDS:-4000}"
export PYTHONUNBUFFERED=1
TRAIN_RATE="${TRAIN_RATE:-0.3}"
MODEL=lgb_v2                       # stage-B scripts read lgb_v2*.txt / *_thresholds.json from WORK
mkdir -p "$WORK" "$OUT"
cd "$HERE/src"
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

WORDS="w_q_max,w_q_min,w_q_n,w_d_max,w_d_min,w_d_n"   # vocabulary-dependent extra-word features
step "A1 TSV -> parquet";                 python io_utils.py "$DATA" "$WORK"
step "A2 learn transliteration";          [ -f "$WORK/translit.json" ] || python translit.py "$WORK" "$WORK/translit.json"
step "A3 normalise records";              python prep.py "$WORK" train,test
step "A4 train/val features";             [ -f "$WORK/feat_val.parquet" ] || python pipeline.py features "$WORK" "$TRAIN_RATE"
step "A5 extra-word table";               [ -f "$WORK/extra_words.json" ] || python augment.py "$WORK" tok
step "A6 stage-1 LightGBM + tune";        [ -f "$WORK/$MODEL.txt" ] || ER_AUG=tok python train.py "$WORK" "$MODEL"
step "A7 stage-1 model without word table (France)"
[ -f "$WORK/${MODEL}_novocab.txt" ] || ER_DROP="$WORDS" python train.py "$WORK" "${MODEL}_novocab"
step "A8 stage-1 scores of all test candidates"
[ -f "$WORK/test_scores.parquet" ] || python pipeline.py predict "$WORK" test "$WORK/$MODEL.txt" "$WORK/${MODEL}_novocab.txt"

step "B  cross-encoder stage"
export ER_V2_WORK="$WORK" ER_RAW="$WORK" ER_CE_WORK="$WORK/ce" ER_DATA="$DATA"
CE_FRANCE=roles bash crossencoder/run_ce_all.sh "$WORK/output_v6a"      # intermediate submission (v6a)

step "C  US/India final matcher (src/v9)"
export ER_CE_STAGE_WORK="$WORK/ce" ER_V9_WORK="$WORK/v9" ER_V10_WORK="$WORK/v10"
export ER_CE_WORK="$WORK/v9/ce4"          # the cross-encoder helpers read ER_CE_WORK
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ER_CE_WORK"
for f in pairs_val.parquet pairs_test.parquet tok_train_v250002_tr.npz tok_test_v250002_tr.npz; do
  ln -sf "$WORK/ce/$f" "$ER_CE_WORK/$f"   # stage-B pair lists and tokenised records
done
cd "$HERE/src/v9"
python ustats.py stats
for s in val train test; do python ustats.py feats $s; done
python make_folds4.py                     # pairs_train.parquet with 4 folds by S1 (country, state)
for f in 0 1 2 3; do
  python ce_model4.py train e5k4_f$f --fold $f --epochs 2 --text tr --eval_every 8000
  python ce_model4.py score e5k4_f$f train
  python ce_model4.py score e5k4_f$f val
  python ce_model4.py score e5k4_f$f test --min_p1 0.01
done
python ea.py block train && python ea.py block test
for s in train val test; do python ea.py ce $s; python ea.py u $s; done
CE4=e5k4_f0+e5k4_f1+e5k4_f2+e5k4_f3
python train9.py k4_u_ea --ce $CE4 --no_w --u --ea
python predict9.py k4_u_ea "$ER_V9_WORK/test_usin_k4_u_ea.parquet" --ce $CE4 --ce_dir "$ER_CE_WORK" --countries US,India --ea

step "D  France veto + CE-confirmed additions (src/v10), final TSVs"
cd "$HERE/src/v10"
python prep_fr.py                         # France rows of the stage-B (v6a) test scores
for f in 0 1 2 3; do python ce_fr10.py train $f; python ce_fr10.py score $f; done
python veto_fr.py                         # report: what the France CE would veto
python make_sub10.py "$OUT" --v 0.2 --classes swap --add 0.6:0.98
