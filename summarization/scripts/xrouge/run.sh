set -e

for dataset in fabbri2021 bhandari2020; do
  python src/xrouge/reformat.py \
    --input-file output/questeval-optimization/${dataset}/predictions.jsonl \
    --output-file output/xrouge/${dataset}/questeval/references.jsonl

  python src/score.py \
    --candidate-file data/${dataset}/summaries.jsonl \
    --output-file output/xrouge/${dataset}/questeval/scores.jsonl \
    --device ${CUDA_VISIBLE_DEVICES} \
    --source-file data/${dataset}/summaries.jsonl \
    --reference-file output/xrouge/${dataset}/questeval/references.jsonl \
    --rouge \
    --bertscore \
    --qaeval \
    --questeval \
    --blanc
done

for dataset in fabbri2021 bhandari2020; do
  python src/xrouge/reformat.py \
    --input-file output/reranking/${dataset}/standard/16/questeval/predictions.jsonl \
    --output-file output/xrouge/${dataset}/questeval-rerank/references.jsonl

  python src/score.py \
    --candidate-file data/${dataset}/summaries.jsonl \
    --output-file output/xrouge/${dataset}/questeval-rerank/scores.jsonl \
    --device ${CUDA_VISIBLE_DEVICES} \
    --source-file data/${dataset}/summaries.jsonl \
    --reference-file output/xrouge/${dataset}/questeval-rerank/references.jsonl \
    --rouge \
    --bertscore \
    --qaeval \
    --questeval \
    --blanc
done


for metric in rouge bertscore qaeval; do
  python src/xrouge/plot.py \
    --standard-dir output/score-submissions \
    --xrouge-dir output/xrouge \
    --ref-metric ${metric} \
    --ref-free-metric questeval \
    --output-file output/xrouge/plots/${metric}-questeval.pdf

  python src/xrouge/plot.py \
    --standard-dir output/score-submissions \
    --xrouge-dir output/xrouge \
    --ref-metric ${metric} \
    --ref-free-metric questeval-rerank \
    --output-file output/xrouge/plots/${metric}-questeval-rerank.pdf
done