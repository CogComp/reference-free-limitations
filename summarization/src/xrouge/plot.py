import argparse
import json
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from glob import glob
from scipy.stats import pearsonr
from typing import Dict, List

DATASETS = ["fabbri2021", "bhandari2020"]
NAMES = {
    "fabbri2021": "SummEval",
    "bhandari2020": "REALSumm",
}


def _get_metrics(metrics: Dict, dataset: str) -> Dict:
    if dataset == "fabbri2021":
        return {
            "rouge": metrics["rouge"]["rouge-2"]["f1"],
            "bertscore": metrics["bertscore"]["f1"] * 100,
            "qaeval": metrics["qaeval"]["f1"] * 100,
            "questeval": metrics["questeval"] * 100 if "questeval" in metrics else None,
            # "blanc": metrics["blanc"],
        }
    elif dataset == "bhandari2020":
        return {
            "rouge": metrics["rouge"]["rouge-2"]["recall"],
            "bertscore": metrics["bertscore"]["recall"] * 100,
            "qaeval": metrics["qaeval"]["f1"] * 100,
            "questeval": metrics["questeval"] * 100 if "questeval" in metrics else None,
            # "blanc": metrics["blanc"],
        }
    else:
        raise Exception()


def _load_standard_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for dataset in DATASETS:
        scores_file = f"{input_dir}/{dataset}.jsonl"
        with open(scores_file, "r") as f:
            for line in f:
                data = json.loads(line)
                system = data["system"]
                scores_dict[dataset][system] = _get_metrics(data["metrics"], dataset)
    return scores_dict


def _load_xrouge_scores(input_dir: str, ref_free_metric: str):
    scores_dict = defaultdict(dict)
    for dataset in DATASETS:
        scores_file = f"{input_dir}/{dataset}/{ref_free_metric}/scores.jsonl"
        with open(scores_file, "r") as f:
            for line in f:
                data = json.loads(line)
                system = data["system"]
                scores_dict[dataset][system] = _get_metrics(data["metrics"], dataset)
    return scores_dict


def _run(
    standard_scores: Dict,
    xbleu_scores: Dict,
    ref_metric: str,
    ref_free_metric: str,
    output_file: str
):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    if ref_metric == "rouge":
        ref_name = "ROUGE-2"
    elif ref_metric == "bertscore":
        ref_name = "BERTScore"
    elif ref_metric == "qaeval":
        ref_name = "QAEval"
    else:
        raise Exception()

    if ref_free_metric == "questeval":
        ref_free_name = "QuestEval"
        opt_name = "Greedy Extractive (\\S\\ref{sec:greedy_ext})"
    elif ref_free_metric == "questeval-rerank":
        ref_free_metric = "questeval"
        ref_free_name = "QuestEval"
        opt_name = "Reranking (\\S\\ref{sec:reranking})"
    elif ref_free_metric == "blanc":
        ref_free_name = "BLANC-Help"
    else:
        raise Exception()

    for i, (dataset, ax) in enumerate(zip(DATASETS, axes)):
        x = []
        y = []
        for system in standard_scores[dataset].keys():
            xbleu = xbleu_scores[dataset][system][ref_metric]
            prism_src = standard_scores[dataset][system][ref_free_metric]
            x.append(xbleu)
            y.append(prism_src)

        df = pd.DataFrame({"x":x, "y":y})
        sns.regplot(x="x", y="y", data=df, ax=ax)

        r = pearsonr(x, y)[0]

        print(f"{ref_free_name} & {opt_name} & {ref_name} & {NAMES[dataset]} & {r:.2f} \\\\")

        ax.title.set_text(f"{NAMES[dataset]} (r={r:.2f})")
        ax.grid()

        ax.set_xlabel(None)
        if i == 0:
            ax.set_ylabel(ref_free_name)
        else:
            ax.set_ylabel(None)

    fig.text(0.5, 0.01, f"{ref_name} using {ref_free_name}'s Pseudo-Reference", ha='center')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_file)


def main(args):
    plt.rcParams.update({'font.size': 14})

    standard_scores = _load_standard_scores(args.standard_dir)
    xrouge_scores = _load_xrouge_scores(args.xrouge_dir, args.ref_free_metric)

    _run(standard_scores, xrouge_scores, args.ref_metric, args.ref_free_metric, args.output_file)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--standard-dir", required=True)
    argp.add_argument("--xrouge-dir", required=True)
    argp.add_argument("--ref-metric", required=True)
    argp.add_argument("--ref-free-metric", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)