#!/bin/bash
#SBATCH --job-name=suswir_referencefree
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/suswir_out_%j.log
#SBATCH --error=logs/suswir_err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}" 

cd ../../evaluation/referencefree

python3 suswir.py --output ./output/alsho_argmatch_preprocessed
