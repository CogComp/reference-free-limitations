import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from matplotlib.lines import Line2D
from typing import Dict, List

LPS = ["de-en", "en-de", "en-ru", "ru-en"]
NAMES = {
    "bleu": "BLEU",
    "bleurt": "BLEURT",
    "prism": "Prism",
    "comet": "COMET",
    "prism-src": "Prism-src",
    "comet-src": "COMET-src"
}


def _load_metrics(
    input_dir: str,
    beam_sizes: List[float],
    reranking_metric: str,
) -> Dict[str, List[float]]:
    metrics = defaultdict(list)

    for size in beam_sizes:
        metrics_dict = json.load(
            open(f"{input_dir}/{size}/{reranking_metric}/scores.json", "r")
        )
        for metric, value in metrics_dict["metrics"].items():
            metrics[metric].append(value)

    return metrics


def _convert_to_relative_change(values: List[float]) -> List[float]:
    start = values[0]
    rel = []
    for val in values:
        diff = val - start
        sign = 1 if diff >= 0 else -1
        rel.append(abs(diff) / abs(start) * sign * 100)
    return rel
    # return [(value - start) / start * 100 for value in values]


def _plot(
    ax,
    lp: str,
    beam_sizes: List[float],
    metrics: Dict[str, List[float]],
    plotting_metrics: List[str],
) -> None:
    for metric in plotting_metrics:
        values = metrics[metric]
        relative_values = _convert_to_relative_change(values)
        ax.plot(beam_sizes, relative_values, label=metric)
    ax.title.set_text(lp)
    ax.grid()


def _run_plot_all(
    input_dir: str,
    method: str,
    reranking_metrics: List[str],
    lps: List[str],
    beam_sizes: List[int],
    plotting_metrics: List[str],
    output_dir: str,
):
    rows = 2
    cols = 2

    for reranking_metric in reranking_metrics:
        fig, axes = plt.subplots(rows, cols)
        row, col = 0, 0
        for lp in lps:
            metrics = _load_metrics(
                f"{input_dir}/{lp}/{method}", beam_sizes, reranking_metric
            )

            _plot(axes[row][col], lp, beam_sizes, metrics, plotting_metrics)

            col += 1
            if col % cols == 0:
                col = 0
                row += 1

        axes[0][0].legend()

        output_file = f"{output_dir}/beam-size/{reranking_metric}/all.pdf"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()


def _plot_pairs(
    ax1,
    ax2,
    beam_sizes: List[float],
    metrics: Dict[str, List[float]],
    ref_based_metric: str,
    ref_free_metric: str,
) -> None:
    values1 = metrics[ref_based_metric]
    values2 = metrics[ref_free_metric]

    ax1.plot(beam_sizes, values1, label=ref_based_metric, color="tab:blue")
    ax2.plot(beam_sizes, values2, label=ref_free_metric, color="tab:orange")


def _run_plot_pairwise(
    input_dir: str,
    method: str,
    reranking_metrics: List[str],
    lps: List[str],
    beam_sizes: List[int],
    ref_based_metrics: List[str],
    ref_free_metrics: List[str],
    output_dir: str,
):
    rows = 2
    cols = 2

    for reranking_metric in reranking_metrics:
        for ref_based in ref_based_metrics:
            for ref_free in ref_free_metrics:

                fig, axes = plt.subplots(rows, cols, sharex=True)
                row, col = 0, 0
                for lp in lps:
                    metrics = _load_metrics(
                        f"{input_dir}/{lp}/{method}", beam_sizes, reranking_metric
                    )

                    ax1 = axes[row][col]
                    ax2 = ax1.twinx()
                    _plot_pairs(ax1, ax2, beam_sizes, metrics, ref_based, ref_free)

                    ax1.title.set_text(lp)
                    ax1.grid()

                    if row == 1:
                        ax1.set_xlabel("Beam Size")
                    if col == 0:
                        ax1.set_ylabel(NAMES[ref_based])
                    if col == 1:
                        ax2.set_ylabel(NAMES[ref_free])

                    col += 1
                    if col % cols == 0:
                        col = 0
                        row += 1

                elements = [
                    Line2D([0], [0], color="tab:blue", linewidth=3, label=NAMES[ref_based]),
                    Line2D([0], [0], color="tab:orange", linewidth=3, label=NAMES[ref_free]),
                ]
                fig.legend(handles=elements, ncol=2, loc="lower center")

                output_file = f"{output_dir}/beam-size/{reranking_metric}/{ref_based}-{ref_free}.pdf"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.17)
                plt.savefig(output_file)
                plt.close()


def main(args):
    lps = ["de-en", "en-de", "ru-en", "en-ru"]
    beam_sizes = [1, 2, 4, 8, 16, 32, 64]
    reranking_metrics = ["standard", "prism", "comet"]
    ref_based_metrics = ["bleu", "bleurt", "prism", "comet"]
    ref_free_metrics = ["prism-src", "comet-src"]

    _run_plot_all(
        args.input_dir,
        args.method,
        reranking_metrics,
        lps,
        beam_sizes,
        ref_based_metrics + ref_free_metrics,
        args.output_dir,
    )

    _run_plot_pairwise(
        args.input_dir,
        args.method,
        reranking_metrics,
        lps,
        beam_sizes,
        ref_based_metrics,
        ref_free_metrics,
        args.output_dir,
    )




if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-dir", required=True)
    argp.add_argument("--output-dir", required=True)
    argp.add_argument("--method", required=True)
    args = argp.parse_args()
    main(args)
