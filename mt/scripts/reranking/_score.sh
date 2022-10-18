language=$1
input_file=$2
pred_file=$3
output_file=$4

python src/reranking/score.py \
  --input-file ${input_file} \
  --pred-file ${pred_file} \
  --devices 0 1 2 3 4 5 6 7 \
  --language ${language} \
  --output-file ${output_file}
