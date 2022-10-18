score_file=$1
standard_file=$2
prism_file=$3
comet_file=$4

python src/reranking/rerank.py \
  --score-file ${score_file} \
  --standard-file ${standard_file} \
  --prism-file ${prism_file} \
  --comet-file ${comet_file}
