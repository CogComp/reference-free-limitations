DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

for lp in "de-en" "en-de" "en-ru" "ru-en"; do
  src=${lp:0:2}
  tgt=${lp:3:5}
  source_file=data/wmt19/wmt19-submitted-data-v3/txt/sources/newstest2019-${src}${tgt}-src.${src}
  reference_file=data/wmt19/wmt19-submitted-data-v3/txt/references/newstest2019-${src}${tgt}-ref.${tgt}

  for beam_size in 1 2 4 8 16 32 64; do
    output_dir=output/reranking/${lp}/standard/${beam_size}
    pred_file=${output_dir}/predictions.jsonl
    score_file=${output_dir}/scores.jsonl
    mkdir -p ${output_dir}

    log_dir=${output_dir}/logs
    log_file=${log_dir}/translate.log
    mkdir -p ${log_dir}

    sbatch --output ${log_file} --job-name ${lp}-translate-standard-${beam_size} \
      ${DIR}/_translate_standard.sh ${lp} ${source_file} ${pred_file} ${beam_size}

    sh ${DIR}/_score.sh ${tgt} ${source_file} ${pred_file} ${score_file}

    sh ${DIR}/_rerank.sh \
      ${score_file} \
      ${output_dir}/standard/predictions.txt \
      ${output_dir}/prism/predictions.txt \
      ${output_dir}/comet/predictions.txt

    for method in standard prism comet; do
      sh ${DIR}/_evaluate.sh \
        ${lp} \
        ${method}-${beam_size} \
        ${source_file} \
        ${reference_file} \
        ${output_dir}/${method}/predictions.txt \
        ${output_dir}/${method}/scores.json
    done
  done
done

python src/reranking/analyze.py \
  --input-dir output/reranking \
  --output-dir output/reranking/results \
  --method standard