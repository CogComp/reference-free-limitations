set -e

for lp in "de-en" "fi-en" "gu-en" "kk-en" "lt-en" "ru-en" "zh-en" "en-cs" "en-de" "en-fi" "en-gu" "en-kk" "en-lt" "en-ru" "en-zh" "de-cs" "de-fr" "fr-de"; do
  src=${lp:0:2}
  tgt=${lp:3:5}
  source_file=data/wmt19/wmt19-submitted-data-v3/txt/sources/newstest2019-${src}${tgt}-src.${src}
  reference_file=data/wmt19/wmt19-submitted-data-v3/txt/references/newstest2019-${src}${tgt}-ref.${tgt}

  output_dir=output/score-wmt/${lp}

  # Score the submissions
  submission_output_dir=${output_dir}/submissions
  mkdir -p ${submission_output_dir}
  for hyp_file in data/wmt19/wmt19-submitted-data-v3/txt/system-outputs/newstest2019/${lp}/*; do
    # Extract the system name from the filename
    filename=$(basename ${hyp_file})
    filename=${filename%.*}
    system=$(echo ${filename} | cut -c 14-)

    output_file=${submission_output_dir}/${system}.json

    echo ${lp} ${system}

    python src/score.py \
      --candidate-file ${hyp_file} \
      --output-file ${output_file} \
      --system-name ${system} \
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

  # Score the reference
  python src/score.py \
    --candidate-file ${reference_file} \
    --output-file ${output_dir}/reference.json \
    --system-name "reference" \
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