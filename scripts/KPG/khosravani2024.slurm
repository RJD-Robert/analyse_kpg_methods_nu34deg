#!/bin/bash
#SBATCH --job-name=arg_khosravani2024
#SBATCH --mem=128g
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=logs/khos/out_%j.log
#SBATCH --error=logs/khos/err_%j.log
#SBATCH --container-image=robertjosefdomogalla/khosravani_cuda:latest

cd "${SLURM_SUBMIT_DIR}"

cd ../../khosravani2024

python3 KPG_pipeline.py