#!/bin/bash
#SBATCH --job-name=trl_summarize
#SBATCH --output=logs/%j/job_output.txt
#SBATCH --error=logs/%j/job_error.txt
#SBATCH --time=8:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

set -e
source mila.sh
# $@

if [[ "$@" == *"sanity_check"* ]]; then
    DEBUG_ARG=" --sanity_check"
else
    DEBUG_ARG=" "
fi

MODEL_PATH=$(readlink -f output_dir)
echo "Using output dir symlinked: $MODEL_PATH"
MODEL_PATH_ARG="--model_name_or_path $MODEL_PATH"
# python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG $DEBUG_ARG

if [[ "$MODEL_PATH" == *"smol135m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/SmolLM2-135M-Instruct_tldr-sft"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi
python load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG $DEBUG_ARG
