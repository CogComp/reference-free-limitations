import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from collections import defaultdict
from glob import glob
from scipy.stats import pearsonr
from typing import Dict, List


def _load_wmt_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)
        for submission_file in glob(f"{lp_dir}/submissions/*"):
            name = os.path.basename(submission_file)[:-5]
            submission = json.load(open(submission_file, "r"))
            scores_dict[lp][name] = submission["metrics"]
            # Hack for bertscore
            scores_dict[lp][name]["bertscore"] = submission["metrics"]["bertscore"]["f1"]
    return scores_dict


def _load_xbleu_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)
        for submission_file in glob(f"{lp_dir}/scores/*"):
            name = os.path.basename(submission_file)[:-5]
            submission = json.load(open(submission_file, "r"))
            scores_dict[lp][name] = submission["metrics"]
            # Hack for bertscore
            scores_dict[lp][name]["bertscore"] = submission["metrics"]["bertscore"]["f1"]
    return scores_dict


def _run(
    wmt_scores: Dict,
    xbleu_scores: Dict,
    ref_metric: str,
    ref_free_metric: str,
    lps: List[str],
    ncols: int,
    output_file: str
):
    assert ncols in [2, 5]
    width = 12 if ncols == 5 else 6

    nrows = math.ceil(len(lps) / ncols)
    height = nrows * 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axes = [ax for row in axes for ax in row]
    i, j = 0, 0

    if ref_metric == "bleu":
        ref_name = "BLEU"
    elif ref_metric == "bleurt":
        ref_name = "BLEURT"
    elif ref_metric == "bertscore":
        ref_name = "BERTScore"
    else:
        raise Exception()

    if ref_free_metric == "prism-src":
        ref_free_name = "Prism-src"
        opt_name = "Direct Optimization (\\S\\ref{sec:direct_opt})"
    elif ref_free_metric == "comet-src":
        ref_free_name = "COMET-QE"
        opt_name = "Reranking (\\S\\ref{sec:reranking})"
    else:
        raise Exception()

    correlations = []

    lps = sorted(lps)
    for lp, ax in zip(lps, axes):
        x = []
        y = []
        for system in wmt_scores[lp].keys():
            xbleu = xbleu_scores[lp][system][ref_metric]
            prism_src = wmt_scores[lp][system][ref_free_metric]
            x.append(xbleu)
            y.append(prism_src)

        df = pd.DataFrame({"x":x, "y":y})
        sns.regplot(x="x", y="y", data=df, ax=ax)

        r = pearsonr(x, y)[0]
        correlations.append(r)

        # print(f"{ref_free_name} & {opt_name} & {ref_name} & {lp} & {r:.2f} \\\\")

        ax.title.set_text(f"{lp} (r={r:.2f})")
        ax.grid()

        is_first_col = j == 0
        is_last_row = i == nrows - 1
        if is_first_col:
            ax.set_ylabel(ref_free_name)
        else:
            ax.set_ylabel(None)

        ax.set_xlabel(None)
        # if is_last_row and nrows != 4:
        #     # ax.set_xlabel(f"{ref_name} against\n{ref_free_name}-as-a-Model")
        #     ax.set_xlabel(f"{ref_name} using {ref_free_name}'s\nPseudo-Reference")
        # else:
        #     ax.set_xlabel(None)

        j = (j + 1) % ncols
        if j == 0:
            i += 1

    print(lps)
    print(len(lps), ref_free_metric, ref_metric, opt_name)
    print(" & ".join([f"{r:.2f}" for r in correlations + [np.mean(correlations)]]))
    print(f"{ref_free_name}, {ref_metric}, {len(lps)}, Average correlation: {np.mean(correlations)}")
    print()


    # Put a common x-axis label
    if nrows == 4:
        fig.text(0.5, 0.01, f"{ref_name} using {ref_free_name}'s Pseudo-Reference", ha='center')
    elif nrows == 2:
        fig.text(0.5, 0.01, f"{ref_name} using {ref_free_name}'s Pseudo-Reference", ha='center')
    else:
        raise Exception()

    # Delete unused plots
    for index in range(len(lps), nrows * ncols):
        fig.delaxes(axes[index])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.tight_layout()
    # if nrows == 2:
    #     plt.subplots_adjust(bottom=0.25)
    # elif nrows == 4:
    #     plt.subplots_adjust(bottom=0.2)
    plt.savefig(output_file)


def main(args):
    plt.rcParams.update({'font.size': 14})

    wmt_scores = _load_wmt_scores(args.wmt_dir)
    xbleu_scores = _load_xbleu_scores(args.xbleu_dir)

    if args.ref_free_metric == "prism-src":
        _run(wmt_scores, xbleu_scores, args.ref_metric, args.ref_free_metric, [
            "de-en",
            "fi-en",
            "kk-en",
            "lt-en",
            "ru-en",
            "en-de",
            "en-fi",
            "en-kk",
            "en-lt",
            "en-ru",
        ], 5, f"{args.output_dir}/{args.ref_metric}.subset.pdf")
        _run(wmt_scores, xbleu_scores, args.ref_metric, args.ref_free_metric, [
            "en-cs",
            "en-de",
            "en-fi",
            "en-kk",
            "en-lt",
            "en-ru",
            "en-zh",
            "de-cs",
            "de-fr",
            "fr-de",
            "de-en",
            "fi-en",
            "kk-en",
            "lt-en",
            "ru-en",
            "zh-en",
        ], 5, f"{args.output_dir}/{args.ref_metric}.all.pdf")
    elif args.ref_free_metric == "comet-src":
        _run(wmt_scores, xbleu_scores, args.ref_metric, args.ref_free_metric, [
            "de-en",
            "ru-en",
            "en-de",
            "en-ru",
        ], 2, f"{args.output_dir}/{args.ref_metric}.subset.pdf")
    elif args.ref_free_metric == "prism-src-rerank":
        _run(wmt_scores, xbleu_scores, args.ref_metric, "prism-src", [
            "de-en",
            "ru-en",
            "en-de",
            "en-ru",
        ], 2, f"{args.output_dir}/{args.ref_metric}.subset.pdf")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--wmt-dir", required=True)
    argp.add_argument("--xbleu-dir", required=True)
    argp.add_argument("--ref-metric", required=True)
    argp.add_argument("--ref-free-metric", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)