lp=$1
system=$2
source_file=$3
reference_file=$4
candidate_file=$5
output_file=$6

python src/score.py \
  --candidate-file ${candidate_file} \
  --output-file ${output_file} \
  --system-name ${system} \
  --lp ${lp} \
  --source-file ${source_file} \
  --reference-file ${reference_file} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --bleu \
  --bleurt \
  --comet \
  --comet-src \
  --prism \
  --prism-src \
  --bertscore
