#!/bin/bash
# One-shot setup + launch. Run from the repo root on giano.
set -e

echo "== creating venv and installing deps (takes a few minutes) =="
python3 -m venv venv
venv/bin/pip3 install --no-cache-dir -q -r requirements.txt || \
venv/bin/pip3 install --no-cache-dir -q tensorflow gymnasium

mkdir -p logs runs

echo "== submitting smoke test =="
SMOKE=$(sbatch --parsable cluster/smoke.sbatch)
echo "smoke job id: $SMOKE"

echo "== queueing main array (starts only if smoke passes) =="
sbatch --dependency=afterok:$SMOKE --kill-on-invalid-dep=yes cluster/run_array.sbatch

squeue --me
echo "Done. Nothing else to do - emails go to Marco."
