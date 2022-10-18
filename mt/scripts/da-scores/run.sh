set -e

python src/da-scores/extract_scores.py \
  --wmt19-dir data/wmt19/wmt19-metrics-task-package \
  --output-dir output/da-scores
