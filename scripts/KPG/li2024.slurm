#!/bin/bash
#SBATCH --job-name=arg_li2024
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=logs/li/out_%j.log
#SBATCH --error=logs/li/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/li_cuda:latest

cd "${SLURM_SUBMIT_DIR}"

cd ../../li2024

python3 KPG_pipeline.py --topicwise \
                        --input_csv ../modified_KPG/output/argmatch_processed_3x300.csv \
                        --kp_out_file ./output/li_keypoints_argmatch_3x300.csv
