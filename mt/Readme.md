## Python Environment
```
conda create -n ref-free-mt python=3.6
conda activate ref-free-mt

pip install repro==0.1.4
pip install torch==1.10.2
pip install fairseq==0.10.2

# Install other dependencies for translation with fairseq
# https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md#example-usage-torchhub
pip install fastBPE sacremoses subword_nmt

# Install other packages that seem to be required by fairseq but
# weren't originally installed
pip install bitarray

# Other dependenices
pip install matplotlib
```


## WMT'19 Dataset
To setup the WMT'19 dataset, run
```
sh data/wmt19/setup.sh
```

## Reproducing
```
# Calculate all of the metric scores on the WMT data
sh scripts/score-wmt/run.sh

# Plot the system rankings with reference-free/-based metrics
sh scripts/ranking-comparison/run.sh

# Calculate the pseudo-reference correlations:
# 1. Run inference with Prism
sh scripts/prism-optimization/run.sh
# 2. Rerank the output of beam search
sh scripts/reranking/run.sh
# 3. Calculate and plot cross-BLEU
sh scripts/xbleu/run.sh
```