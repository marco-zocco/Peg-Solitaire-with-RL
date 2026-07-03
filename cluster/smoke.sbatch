#!/bin/bash
#SBATCH --job-name=peg-smoke
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.zoccoramazzo@studio.unibo.it
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=rtx2080
#SBATCH --output=logs/smoke_%j.out
#SBATCH --gres=gpu:1
set -e

echo "phase 1 start: $(date)"
venv/bin/python3 -m peg_solitaire.agent --seed 0 --branch-prob 0.2 \
    --episodes 300 --checkpoint-every 100 --outdir runs/smoke
echo "phase 1 end:   $(date)"
venv/bin/python3 -m peg_solitaire.agent --seed 0 --branch-prob 0.2 \
    --episodes 600 --checkpoint-every 100 --outdir runs/smoke
echo "phase 2 end:   $(date)"
