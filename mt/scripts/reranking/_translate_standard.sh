#!/bin/bash
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --constraint=48GBgpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB

lp=$1
input_file=$2
output_file=$3
beam_size=$4

mkdir -p $(dirname ${output_file})

python src/reranking/translate.py \
  --lp ${lp} \
  --input-file ${input_file} \
  --beam-size ${beam_size} \
  --inference-method standard \
  --output-file ${output_file}
