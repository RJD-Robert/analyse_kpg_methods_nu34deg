#!/bin/bash
#SBATCH --job-name=arg_alshomary2021
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=logs/alsho/out_%j.log
#SBATCH --error=logs/alsho/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/alshomary2021:latest

cd "${SLURM_SUBMIT_DIR}"

cd ../../alshomary2021

python3  KPG_pipeline.py --arguments-csv ../modified_KPG/output/argmatch_processed_3x300.csv \
                         --output-path ./output/alshomary_keypoints_argmatch_3x300.csv
