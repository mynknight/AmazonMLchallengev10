#!/usr/bin/env bash
# US/India final model with the hard-pair CEs (e5hk), then test scoring
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mfa
export ER_CE_WORK=/home/mayankc/MLchallenge_v9/work/ce4 ER_THREADS=16 OMP_WAIT_POLICY=passive PYTHONUNBUFFERED=1
cd "$(dirname "$0")/src"
python hk_run.py train
python hk_run.py predict
echo "HK DONE"
