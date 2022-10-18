DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

for dataset in "fabbri2021" "bhandari2020"; do
  input_file=data/${dataset}/summaries.jsonl

  for beam_size in 1 2 4 8 16; do
    output_dir=output/reranking/${dataset}/standard/${beam_size}
    pred_file=${output_dir}/predictions.jsonl
    score_file=${output_dir}/scores.jsonl
    mkdir -p ${output_dir}

    python src/reranking/summarize.py \
      --input-file ${input_file} \
      --beam-size ${beam_size} \
      --output-file ${pred_file} \
      --device ${CUDA_VISIBLE_DEVICES}

    python src/reranking/score.py \
      --input-file ${input_file} \
      --pred-file ${pred_file} \
      --device ${CUDA_VISIBLE_DEVICES} \
      --output-file ${score_file}

    python src/reranking/rerank.py \
      --score-file ${score_file} \
      --standard-file ${output_dir}/standard/predictions.jsonl \
      --questeval-file ${output_dir}/questeval/predictions.jsonl \
      --blanc-file ${output_dir}/blanc/predictions.jsonl

    for method in standard questeval blanc; do
      python src/score.py \
        --candidate-file ${output_dir}/${method}/predictions.jsonl \
        --output-file ${output_dir}/${method}/scores.json \
        --device ${CUDA_VISIBLE_DEVICES} \
        --source-file data/${dataset}/summaries.jsonl \
        --reference-file data/${dataset}/summaries.jsonl \
        --rouge \
        --bertscore \
        --qaeval \
        --questeval \
        --blanc
    done
  done
done

python src/reranking/analyze.py \
  --input-dir output/reranking \
  --output-dir output/reranking/results \
  --method standard
