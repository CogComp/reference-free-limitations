set -e

for metric in bleu bleurt bertscore; do
  python src/ranking-comparison/plot.py \
    --wmt-dir output/score-wmt \
    --opt-method prism \
    --opt-metric-dir output/prism-optimization \
    --ref-based-metric ${metric} \
    --ref-free-metric prism-src \
    --output-dir output/ranking-comparison/${metric}/prism-src

  python src/ranking-comparison/plot.py \
    --wmt-dir output/score-wmt \
    --opt-method comet-rerank \
    --opt-metric-dir output/reranking \
    --ref-based-metric ${metric} \
    --ref-free-metric comet-src \
    --output-dir output/ranking-comparison/${metric}/comet-src

  python src/ranking-comparison/plot.py \
    --wmt-dir output/score-wmt \
    --opt-method prism-rerank \
    --opt-metric-dir output/reranking \
    --ref-based-metric ${metric} \
    --ref-free-metric prism-src \
    --output-dir output/ranking-comparison/${metric}/prism-src
done

# Rename for overleaf
mkdir -p output/ranking-comparison/overleaf

for metric in bleu bleurt bertscore; do
  cp output/ranking-comparison/${metric}/comet-src/comet-rerank.pdf output/ranking-comparison/overleaf/comet-${metric}.pdf
  cp output/ranking-comparison/${metric}/prism-src/prism.all.pdf output/ranking-comparison/overleaf/prism-${metric}.all.pdf
  cp output/ranking-comparison/${metric}/prism-src/prism.subset.pdf output/ranking-comparison/overleaf/prism-${metric}.subset.pdf
  cp output/ranking-comparison/${metric}/prism-src/prism-rerank.pdf output/ranking-comparison/overleaf/prism_rerank-${metric}.pdf
done
