set -e

for dataset in "fabbri2021" "bhandari2020"; do
  output_dir=output/questeval-optimization/${dataset}

  python src/questeval-optimization/optimize.py \
    --input-file data/${dataset}/summaries.jsonl \
    --num-sents 3 \
    --output-file ${output_dir}/predictions.jsonl

  python src/score.py \
    --candidate-file ${output_dir}/predictions.jsonl \
    --output-file ${output_dir}/scores.json \
    --device ${CUDA_VISIBLE_DEVICES} \
    --source-file data/${dataset}/summaries.jsonl \
    --reference-file data/${dataset}/summaries.jsonl \
    --rouge \
    --bertscore \
    --qaeval \
    --questeval \
    --blanc
done
