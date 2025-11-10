#!/bin/bash
#SBATCH --job-name=rf_referencefree
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/rf_out_%j.log
#SBATCH --error=logs/rf_err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

cd "${SLURM_SUBMIT_DIR}" 

cd ../../evaluation/referencefree

python3 rf.py --csv ../../data/evaluation/referencefree/IBM_groundtruth_kp.csv \
               ../../data/evaluation/referencefree/IBM_one_kp_multiple_times.csv \
               ../../data/evaluation/referencefree/IBM_one_sentence.csv \
               ../../data/evaluation/referencefree/IBM_random_sentences.csv \
               ../../data/evaluation/referencefree/IBM_same_kp_reformulated.csv \
               ../../data/evaluation/referencefree/IBM_topics_swapped.csv \
               ../../alshomary2021/results/alshomary_keypoints_IBM.csv \
               ../../khosravani2024/results/khosravani_keypoints_IBM.csv \
               ../../li2024/results/li_keypoints_IBM.csv  \
               --out-prefix rf_all_results \
              --verbose


# python3 rf.py --csv ../../data/evaluation/referencefree/IBM_groundtruth_kp.csv \
#                 --out-prefix rf_gt_results \
#                 --verbose
