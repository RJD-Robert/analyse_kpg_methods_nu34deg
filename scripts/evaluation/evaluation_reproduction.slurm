#!/bin/bash
#SBATCH --job-name=eval_reproduction
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval/out_%j.log
#SBATCH --error=logs/eval/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}"

cd ../../evaluation/reproduction

python3 paper_evaluation.py --system_csv ../../khosravani2024/results/khosravani_keypoints_IBM_LLM_evaluated.csv \
                            --reference_csv ../../data/evaluation/referencefree/IBM_groundtruth_kp.csv
