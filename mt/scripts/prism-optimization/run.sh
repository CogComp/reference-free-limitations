set -e

for lp in "de-en" "fi-en" "kk-en" "lt-en" "ru-en" "zh-en" "en-cs" "en-de" "en-fi" "en-kk" "en-lt" "en-ru" "en-zh" "de-cs" "de-fr" "fr-de"; do
  src=${lp:0:2}
  tgt=${lp:3:5}
  source_file=data/wmt19/wmt19-submitted-data-v3/txt/sources/newstest2019-${src}${tgt}-src.${src}
  reference_file=data/wmt19/wmt19-submitted-data-v3/txt/references/newstest2019-${src}${tgt}-ref.${tgt}

  output_dir=output/prism-optimization/${lp}
  pred_file=${output_dir}/predictions.txt
  score_file=${output_dir}/scores.json
  mkdir -p ${output_dir}

  python src/prism-optimization/translate.py \
    --input-file ${source_file} \
    --language ${tgt} \
    --device ${CUDA_VISIBLE_DEVICES} \
    --output-file ${pred_file}

  python src/score.py \
    --candidate-file ${pred_file} \
    --output-file ${score_file} \
    --system-name "prism" \
    --lp ${lp} \
    --source-file ${source_file} \
    --reference-file ${reference_file} \
    --device ${CUDA_VISIBLE_DEVICES} \
    --bleu \
    --bleurt \
    --comet \
    --comet-src \
    --prism \
    --prism-src \
    --bertscore
done