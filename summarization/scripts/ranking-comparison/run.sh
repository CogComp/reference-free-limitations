set -e

for metric in rouge bertscore qaeval; do
  python src/ranking-comparison/plot.py \
    --submissions-dir output/score-submissions \
    --opt-method questeval \
    --opt-method-dir output/questeval-optimization \
    --ref-based-metric ${metric} \
    --ref-free-metric questeval \
    --output-dir output/ranking-comparison/${metric}/questeval

  python src/ranking-comparison/plot.py \
    --submissions-dir output/score-submissions \
    --opt-method questeval-rerank \
    --opt-method-dir output/reranking \
    --ref-based-metric ${metric} \
    --ref-free-metric questeval \
    --output-dir output/ranking-comparison/${metric}/questeval

  python src/ranking-comparison/plot.py \
    --submissions-dir output/score-submissions \
    --opt-method blanc-rerank \
    --opt-method-dir output/reranking \
    --ref-based-metric ${metric} \
    --ref-free-metric blanc \
    --output-dir output/ranking-comparison/${metric}/blanc
done

mkdir -p output/ranking-comparison/overleaf
for metric in rouge bertscore qaeval; do
  cp output/ranking-comparison/${metric}/questeval/questeval.pdf output/ranking-comparison/overleaf/questeval-${metric}.pdf
  cp output/ranking-comparison/${metric}/questeval/questeval-rerank.pdf output/ranking-comparison/overleaf/questeval_rerank-${metric}.pdf
done