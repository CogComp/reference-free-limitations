import argparse
import json
import math
import matplotlib.pyplot as plt
import os
from collections import defaultdict, namedtuple
from glob import glob
from typing import Dict, List


def _load_wmt_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)
        for submission_file in glob(f"{lp_dir}/submissions/*"):
            name = os.path.basename(submission_file)[:-5]
            submission = json.load(open(submission_file, "r"))
            scores_dict[lp][name] = submission["metrics"]
    return scores_dict


def _load_xbleu_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)
        for submission_file in glob(f"{lp_dir}/scores/*"):
            name = os.path.basename(submission_file)[:-5]
            submission = json.load(open(submission_file, "r"))
            scores_dict[lp][name] = submission["metrics"]
    return scores_dict


def _run(
    wmt_scores: Dict,
    xbleu_scores: Dict,
    metric: str,
    lps: List[str],
    output_file: str
):
    ncols = 5
    width = 12

    nrows = math.ceil(len(lps) / ncols)
    height = nrows * 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), sharey=True, sharex=True)
    axes = [ax for row in axes for ax in row]
    i, j = 0, 0

    for lp, ax in zip(lps, axes):
        tuples = []
        for system in wmt_scores[lp].keys():
            bleu = wmt_scores[lp][system][metric]
            prism_src = wmt_scores[lp][system]["prism-src"]
            xbleu = xbleu_scores[lp][system][metric]
            tuples.append((xbleu, bleu, prism_src))

        # Sort by bleu to get bleu ranks
        tuples.sort(key=lambda t: t[1], reverse=True)
        tuples = [(*tup, i) for i, tup in enumerate(tuples)]

        # Sort by prism_src to get prism-src ranks
        tuples.sort(key=lambda t: t[2], reverse=True)
        tuples = [(*tup, i) for i, tup in enumerate(tuples)]

        # x axis is the xbleu score (similarity)
        # y axis is the diff between the ranks. A positive result
        #     means prism ranked it higher
        x = [tup[0] for tup in tuples]
        # y = [tup[3] - tup[4] for tup in tuples]
        y = [tup[1] for tup in tuples]

        ax.scatter(x, y)
        ax.title.set_text(lp)
        ax.grid()

    # Delete unused plots
    for index in range(len(lps), nrows * ncols):
        fig.delaxes(axes[index])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.tight_layout()
    if nrows == 2:
        plt.subplots_adjust(bottom=0.15)
    elif nrows == 4:
        plt.subplots_adjust(bottom=0.09)
    plt.savefig(output_file)


def main(args):
    wmt_scores = _load_wmt_scores(args.wmt_dir)
    xbleu_scores = _load_xbleu_scores(args.xbleu_dir)

    _run(wmt_scores, xbleu_scores, args.metric, [
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
            ], f"{args.output_dir}/{args.metric}.subset.pdf")



if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--wmt-dir", required=True)
    argp.add_argument("--xbleu-dir", required=True)
    argp.add_argument("--metric", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)