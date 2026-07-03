#!/bin/bash
#SBATCH --job-name=peg-final
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.zoccoramazzo@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=rtx2080
#SBATCH --array=0-3
#SBATCH --output=logs/final_%a.out
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1

SEED=$((SLURM_ARRAY_TASK_ID % 2 + 1))
if [ $SLURM_ARRAY_TASK_ID -lt 2 ]; then BP=0.0; else BP=0.2; fi

venv/bin/python3 -m peg_solitaire.agent \
    --seed $SEED --episodes 12000 --branch-prob $BP \
    --checkpoint-every 200 --outdir runs/bp${BP}_seed${SEED}
