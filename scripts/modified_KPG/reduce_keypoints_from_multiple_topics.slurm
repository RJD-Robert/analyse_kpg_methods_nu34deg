#!/bin/bash
#SBATCH --job-name=reduce_keypoints_from_multiple_topics
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/eval/out_%j.log
#SBATCH --error=logs/eval/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}" 

cd ../../modified_KPG

python3 reduce_keypoints_from_multiple_topics.py \
  --input ../li2024/results/li_keypoints_argmatch_3x300.csv \
  --output ../li2024/results/li_keypoints_argmatch_3x300_reduced.csv \


