set -e

for dataset in "fabbri2021" "bhandari2020"; do
  output_dir=output/score-submissions
  mkdir -p ${output_dir}

  python src/score.py \
    --candidate-file data/${dataset}/summaries.jsonl \
    --output-file ${output_dir}/${dataset}.jsonl \
    --device ${CUDA_VISIBLE_DEVICES} \
    --source-file data/${dataset}/summaries.jsonl \
    --reference-file data/${dataset}/summaries.jsonl \
    --rouge \
    --bertscore \
    --qaeval \
    --questeval \
    --blanc

  python src/score-submissions/extract_references.py \
    --input-file data/${dataset}/summaries.jsonl \
    --output-file ${output_dir}/${dataset}-references.jsonl

  python src/score.py \
    --candidate-file ${output_dir}/${dataset}-references.jsonl \
    --output-file ${output_dir}/${dataset}-references-scores.json \
    --device ${CUDA_VISIBLE_DEVICES} \
    --source-file data/${dataset}/summaries.jsonl \
    --reference-file data/${dataset}/summaries.jsonl \
    --rouge \
    --bertscore \
    --qaeval \
    --questeval \
    --blanc
done