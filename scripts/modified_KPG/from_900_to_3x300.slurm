#!/bin/bash
#SBATCH --job-name=fromm_900_to_3x300
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval/from_%j.log
#SBATCH --error=logs/eval/from_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}" 

cd ../../modified_KPG

python3 from_900_to_3x300.py \
  --input ./output/argmatch_processed_900.csv \
  --output ./output/argmatch_processed_3x300.csv \


