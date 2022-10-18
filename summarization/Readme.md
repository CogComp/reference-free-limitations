## Python Environment
```
conda create -n ref-free-summ python=3.6

pip install repro==0.1.4
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install fairseq==0.10.2

# Install other dependencies for translation with fairseq
# https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md#example-usage-torchhub
pip install fastBPE sacremoses subword_nmt

# Install other packages that seem to be required by fairseq but
# weren't originally installed
pip install bitarray

# Install QuestEval
git clone https://github.com/ThomasScialom/QuestEval
cd QuestEval
git checkout v0.1.1
pip install .

cd QuestEval/unilm/s2s-ft
pip install --no-cache-dir .

pip install nltk

pip install seaborn
```

## Reproducing
```
# Setup the two datasets using the dataset readers from
# https://github.com/danieldeutsch/sacrerouge. The output
# should be data/{fabbri2021,bhandari2020}/summaries.jsonl

# Calculate all of the metric scores on the summarization data
sh scripts/score-submissions/run.sh

# Plot the system rankings with reference-free/-based metrics
sh scripts/ranking-comparison/run.sh

# Calculate the pseudo-reference correlations:
# 1. Run inference with QuestEval
sh scripts/questeval-optimization/run.sh
# 2. Rerank the output of beam search with QuestEval
sh scripts/reranking/run.sh
# 3. Calculate and plot cross-ROUGE
sh scripts/xrouge/run.sh
```