set -e

for lp in "de-en" "fi-en" "kk-en" "lt-en" "ru-en" "zh-en" "en-cs" "en-de" "en-fi" "en-kk" "en-lt" "en-ru" "en-zh" "de-cs" "de-fr" "fr-de"; do
  output_dir=output/xbleu/prism-src/${lp}
  for hyp_file in data/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019/${lp}/*; do
    # Extract the system name from the filename
    filename=$(basename ${hyp_file})
    filename=${filename%.*}
    system=$(echo ${filename} | cut -c 14-)

    # Score using the output from optimizing Prism-src
    # as the reference
    python src/score.py \
      --candidate-file ${hyp_file} \
      --output-file ${output_dir}/scores/${system}.json \
      --system-name ${system} \
      --lp ${lp} \
      --reference-file output/prism-optimization/${lp}/predictions.txt \
      --device ${CUDA_VISIBLE_DEVICES} \
      --bleu \
      --bleurt \
      --bertscore
  done
done

for metric in bleu bleurt bertscore; do
  python src/xbleu/plot_xbleu_similarity.py \
    --wmt-dir output/score-wmt \
    --xbleu-dir output/xbleu/prism-src \
    --ref-metric ${metric} \
    --ref-free-metric prism-src \
    --output-dir output/xbleu/results/prism-src
done


for lp in "de-en" "en-de" "en-ru" "ru-en"; do
  output_dir=output/xbleu/comet-src/${lp}
  for hyp_file in data/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019/${lp}/*; do
    # Extract the system name from the filename
    filename=$(basename ${hyp_file})
    filename=${filename%.*}
    system=$(echo ${filename} | cut -c 14-)

    # Score using the output from optimizing COMET-src
    # as the reference
    python src/score.py \
      --candidate-file ${hyp_file} \
      --output-file ${output_dir}/scores/${system}.json \
      --system-name ${system} \
      --lp ${lp} \
      --reference-file output/reranking/${lp}/standard/64/comet/predictions.txt \
      --device ${CUDA_VISIBLE_DEVICES} \
      --bleu \
      --bleurt \
      --bertscore
    done
done

for metric in bleu bleurt bertscore; do
  python src/xbleu/plot_xbleu_similarity.py \
    --wmt-dir output/score-wmt \
    --xbleu-dir output/xbleu/comet-src \
    --ref-metric ${metric} \
    --ref-free-metric comet-src \
    --output-dir output/xbleu/results/comet-src
done


for lp in "de-en" "en-de" "en-ru" "ru-en"; do
  output_dir=output/xbleu/prism-src-rerank/${lp}
  for hyp_file in data/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019/${lp}/*; do
    # Extract the system name from the filename
    filename=$(basename ${hyp_file})
    filename=${filename%.*}
    system=$(echo ${filename} | cut -c 14-)

    # Score using the output from optimizing Prism-src
    # as the reference using reranking
    python src/score.py \
      --candidate-file ${hyp_file} \
      --output-file ${output_dir}/scores/${system}.json \
      --system-name ${system} \
      --lp ${lp} \
      --reference-file output/reranking/${lp}/standard/64/prism/predictions.txt \
      --device ${CUDA_VISIBLE_DEVICES} \
      --bleu \
      --bleurt \
      --bertscore
    done
done

for metric in bleu bleurt bertscore; do
  python src/xbleu/plot_xbleu_similarity.py \
    --wmt-dir output/score-wmt \
    --xbleu-dir output/xbleu/prism-src-rerank \
    --ref-metric ${metric} \
    --ref-free-metric prism-src-rerank \
    --output-dir output/xbleu/results/prism-src-rerank
done

# Rename for Overleaf
mkdir -p output/xbleu/results/overleaf
cp output/xbleu/results/comet-src/bleu.subset.pdf output/xbleu/results/overleaf/bleu-comet.pdf
cp output/xbleu/results/comet-src/bleurt.subset.pdf output/xbleu/results/overleaf/bleurt-comet.pdf
cp output/xbleu/results/comet-src/bertscore.subset.pdf output/xbleu/results/overleaf/bertscore-comet.pdf

cp output/xbleu/results/prism-src/bleu.subset.pdf output/xbleu/results/overleaf/bleu-prism.subset.pdf
cp output/xbleu/results/prism-src/bleurt.subset.pdf output/xbleu/results/overleaf/bleurt-prism.subset.pdf
cp output/xbleu/results/prism-src/bertscore.subset.pdf output/xbleu/results/overleaf/bertscore-prism.subset.pdf
cp output/xbleu/results/prism-src/bleu.all.pdf output/xbleu/results/overleaf/bleu-prism.all.pdf
cp output/xbleu/results/prism-src/bleurt.all.pdf output/xbleu/results/overleaf/bleurt-prism.all.pdf
cp output/xbleu/results/prism-src/bertscore.all.pdf output/xbleu/results/overleaf/bertscore-prism.all.pdf
