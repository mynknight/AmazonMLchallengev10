#!/usr/bin/env bash
# France-adapted cross-encoders from v6a pseudo-labels (4 folds, GPU)
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mfa
export ER_THREADS=16 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_WAIT_POLICY=passive
export HF_HOME=/home/mayankc/.cache/huggingface HF_HUB_OFFLINE=1
export ER_CE_WORK=/home/mayankc/MLchallenge_v9/work/ce4
cd "$(dirname "$0")/src"
for f in 0 1 2 3; do
  python ce_fr10.py train $f
  python ce_fr10.py score $f
done
echo "CE_FR10 DONE"
