#!/bin/bash
#SBATCH --job-name=referencefree_all
#SBATCH --mem=64g
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/new/ref_free_out_%j.log
#SBATCH --error=logs/new/ref_free_err_%j.log
#SBATCH --container-image=robertjosefdomogalla/paper_evaluation_cuda:latest

# Aufruf: sbatch referencefree_all.sbatch <ARGUMENTS_FILE> <KEYPOINTS_FILE>
# Beispiel: sbatch referencefree_all.sbatch ../../data/ibm/IBM_arguments.csv ../../data/evaluation/referencefree/IBM_groundtruth_kp.csv

set -e  # Stoppe bei Fehler
set -u  # Fehler bei nicht definierten Variablen

ARGS_FILE=$1
KP_FILE=$2
OUTPUT=$3

cd "${SLURM_SUBMIT_DIR}"
cd ../../evaluation/referencefree

echo "Starte reference-free Evaluierung..."
echo "Arguments: ${ARGS_FILE}"
echo "Keypoints: ${KP_FILE}"
echo "Output: ${OUTPUT}"

# AC
python3 ac.py --arguments "${ARGS_FILE}" \
              --keypoints "${KP_FILE}" \
              --output "./${OUTPUT}/ac_all_results.csv"

# IQS
python3 iqs.py --keypoints "${KP_FILE}" \
              --output "./${OUTPUT}/iqs_all_results.csv" \
              --debug-iqs

# MMS
python3 mms.py --arguments "${ARGS_FILE}" \
              --keypoints "${KP_FILE}" \
              --output "./${OUTPUT}/mms_all_results.csv"

# RF
python3 rf.py --csv "${KP_FILE}" \
              --out-prefix rf_all_results \
              --output "./${OUTPUT}" \
              --verbose

# SUSWIR (hat keine Arguments/Keypoints Inputs)
python3 suswir.py --output "./${OUTPUT}"

echo "Alle Evaluierungen abgeschlossen."