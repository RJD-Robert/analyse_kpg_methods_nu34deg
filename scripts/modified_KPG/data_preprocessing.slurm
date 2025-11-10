#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval/out_%j.log
#SBATCH --error=logs/eval/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}" 

cd ../Modified_KPG

python3 data_preprocessing.py \
  --csv ../data/aqlmatch/aqlmatch_arguments.csv \
  --out ./output/arguments_processed.csv \
  --top-k 300 \
  --keep-score


