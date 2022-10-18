import argparse
import json
import math
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from glob import glob
from matplotlib.lines import Line2D
from typing import Dict, List


def _load_wmt_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)

        reference = json.load(open(f"{lp_dir}/reference.json", "r"))
        scores_dict[lp]["reference"] = reference["metrics"]
        # Hack for bertscore
        scores_dict[lp]["reference"]["bertscore"] = reference["metrics"]["bertscore"]["f1"]

        for submission_file in glob(f"{lp_dir}/submissions/*"):
            name = os.path.basename(submission_file)[:-5]
            submission = json.load(open(submission_file, "r"))
            scores_dict[lp][name] = submission["metrics"]
            # Hack for bertscore
            scores_dict[lp][name]["bertscore"] = submission["metrics"]["bertscore"]["f1"]

    return scores_dict


def _load_prism_scores(input_dir: str):
    scores_dict = {}
    for lp_dir in glob(f"{input_dir}/*"):
        lp = os.path.basename(lp_dir)
        scores = json.load(open(f"{lp_dir}/scores.json", "r"))
        scores_dict[lp] = scores["metrics"]
        # Hack for bertscore
        scores_dict[lp]["bertscore"] = scores["metrics"]["bertscore"]["f1"]
    return scores_dict


def _load_rerank_scores(input_dir: str, lps: List[str], metric: str):
    scores_dict = {}
    for lp in lps:
        scores = json.load(
            open(f"{input_dir}/{lp}/standard/64/{metric}/scores.json", "r")
        )
        scores_dict[lp] = scores["metrics"]
        # Hack for bertscore
        scores_dict[lp]["bertscore"] = scores["metrics"]["bertscore"]["f1"]
    return scores_dict


def _rescale_values(x: List[float]) -> List[float]:
    min_x = min(x)
    max_x = max(x)
    return [(value - min_x) / (max_x - min_x) for value in x]


def _plot(
    lps: List[str],
    wmt_scores: Dict,
    opt_scores: Dict,
    ref_metric: str,
    ref_free_metric: str,
    opt_method: str,
    ncols: int,
    output_file: str,
):
    assert ncols in [2, 5]
    width = 12 if ncols == 5 else 6

    nrows = math.ceil(len(lps) / ncols)
    height = nrows * 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axes = [ax for row in axes for ax in row]
    i, j = 0, 0

    for lp, ax1 in zip(lps, axes):
        ax2 = ax1.twinx()

        x_values = []
        y_values = []

        x_values.append(opt_scores[lp][ref_metric])
        y_values.append(opt_scores[lp][ref_free_metric])

        for system, scores in wmt_scores[lp].items():
            if system == "reference":
                continue
            x_values.append(scores[ref_metric])
            y_values.append(scores[ref_free_metric])

        y_ref_value = wmt_scores[lp]["reference"][ref_free_metric]

        x_rescaled = _rescale_values(x_values)
        y_rescaled_with_ref = _rescale_values(y_values + [y_ref_value])
        y_rescaled = y_rescaled_with_ref[:-1]
        y_ref_rescaled = y_rescaled_with_ref[-1]

        for index, (x, y, x_rescale, y_rescale) in enumerate(
            zip(x_values, y_values, x_rescaled, y_rescaled)
        ):
            # print(x, y, x_rescale, y_rescale)

            if index == 0:
                color = "tab:blue"
                width = 2
            else:
                color = "0.8"
                width = 1
            ax1.plot([0, 1], [x_rescale, y_rescale], linewidth=width, color=color)

            # Replot on ax2 but invisible so the y axis is scaled properly
            ax2.plot([0, 1], [x_rescale, y_rescale], color=color, alpha=0.0)

        ax1.scatter(
            [1], [y_ref_rescaled], color="tab:red", s=100, linewidth=2, marker="x"
        )
        # Redo on ax2 but invisible
        ax2.scatter([1], [y_ref_rescaled], color="tab:red", marker="x", alpha=0.0)

        ticks = [0.0, 0.33, 0.66, 1.0]
        min_x = min(x_values)
        min_y = min(y_values)

        x_delta = max(x_values) - min_x
        y_delta = max(y_values + [y_ref_value]) - min_y

        xlabels = []
        ylabels = []
        for tick in ticks:
            value = min_x + x_delta * tick
            xlabels.append(f"{value:.1f}")

            value = min_y + y_delta * tick
            ylabels.append(f"{value:.1f}")

        ax1.set_yticks(ticks)
        ax1.set_yticklabels(xlabels)
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(ylabels)

        ax1.set_xticks([])
        ax1.title.set_text(lp)

        is_first_col = j == 0
        is_last_col = j == ncols - 1
        if is_first_col:
            if ref_metric == "bleu":
                ax1.set_ylabel("BLEU")
            elif ref_metric == "bleurt":
                ax1.set_ylabel("BLEURT")
            elif ref_metric == "bertscore":
                ax1.set_ylabel("BERTScore")
            else:
                raise Exception()
        if is_last_col:
            if ref_free_metric == "prism-src":
                ax2.set_ylabel("Prism-src")
            elif ref_free_metric == "comet-src":
                ax2.set_ylabel("COMET-QE")
            else:
                raise Exception()

        j = (j + 1) % ncols
        if j == 0:
            i += 1

    # Delete unused plots
    for index in range(len(lps), nrows * ncols):
        fig.delaxes(axes[index])

    # Add a legend
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    if opt_method == "prism":
        opt_name = "Prism-src as a Model"
    elif opt_method == "prism-rerank":
        opt_name = "Prism-src as a Model"
    elif opt_method == "comet-rerank":
        opt_name = "COMET-QE as a Model"
    else:
        raise Exception()

    elements = [
        Line2D([0], [0], color="tab:blue", linewidth=3, label=opt_name),
        Line2D([0], [0], color="0.8", linewidth=3, label="WMT'19 Submission"),
        Line2D(
            [0],
            [0],
            color="tab:red",
            marker="x",
            markeredgewidth=3,
            markersize=10,
            linestyle="None",
            label="Reference Translation",
        ),
    ]
    # fig.legend(handles=elements, ncol=3, bbox_to_anchor=(0.5, -0.05))
    if ncols == 5:
        fig.legend(handles=elements, ncol=3, loc="lower center")
    else:
        fig.legend(handles=elements, ncol=2, loc="lower center")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.tight_layout()
    if nrows == 2:
        if ncols == 5:
            plt.subplots_adjust(bottom=0.15)
        else:
            plt.subplots_adjust(bottom=0.25)
    elif nrows == 4:
        plt.subplots_adjust(bottom=0.09)
    plt.savefig(output_file)


def main(args):
    plt.rcParams.update({'font.size': 14})

    wmt_scores = _load_wmt_scores(args.wmt_dir)

    if args.opt_method == "prism":
        prism_scores = _load_prism_scores(args.opt_metric_dir)
        _plot(
            [
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
            ],
            wmt_scores,
            prism_scores,
            args.ref_based_metric,
            args.ref_free_metric,
            args.opt_method,
            5,
            f"{args.output_dir}/prism.subset.pdf",
        )
        _plot(
            [
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
            ],
            wmt_scores,
            prism_scores,
            args.ref_based_metric,
            args.ref_free_metric,
            args.opt_method,
            5,
            f"{args.output_dir}/prism.all.pdf",
        )

    elif args.opt_method == "comet-rerank":
        lps = ["de-en", "ru-en", "en-de", "en-ru"]
        comet_scores = _load_rerank_scores(args.opt_metric_dir, lps, "comet")
        _plot(
            lps,
            wmt_scores,
            comet_scores,
            args.ref_based_metric,
            args.ref_free_metric,
            args.opt_method,
            2,
            f"{args.output_dir}/comet-rerank.pdf",
        )

    elif args.opt_method == "prism-rerank":
        lps = ["de-en", "ru-en", "en-de", "en-ru"]
        comet_scores = _load_rerank_scores(args.opt_metric_dir, lps, "prism")
        _plot(
            lps,
            wmt_scores,
            comet_scores,
            args.ref_based_metric,
            args.ref_free_metric,
            args.opt_method,
            2,
            f"{args.output_dir}/prism-rerank.pdf",
        )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--wmt-dir", required=True)
    argp.add_argument(
        "--opt-method", required=True, choices=["prism", "prism-rerank", "comet-rerank"]
    )
    argp.add_argument("--opt-metric-dir", required=True)
    argp.add_argument("--ref-based-metric", required=True)
    argp.add_argument("--ref-free-metric", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)
