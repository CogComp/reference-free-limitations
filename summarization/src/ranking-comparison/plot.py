import argparse
import json
import math
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from glob import glob
from matplotlib.lines import Line2D
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
            "questeval": metrics["questeval"] * 100,
            "blanc": metrics["blanc"],
        }
    elif dataset == "bhandari2020":
        return {
            "rouge": metrics["rouge"]["rouge-2"]["recall"],
            "bertscore": metrics["bertscore"]["recall"] * 100,
            "qaeval": metrics["qaeval"]["f1"] * 100,
            "questeval": metrics["questeval"] * 100,
            "blanc": metrics["blanc"],
        }
    else:
        raise Exception()


def _load_submission_scores(input_dir: str):
    scores_dict = defaultdict(dict)
    for dataset in DATASETS:
        dataset_file = f"{input_dir}/{dataset}.jsonl"

        with open(dataset_file, "r") as f:
            for line in f:
                data = json.loads(line)
                system = data["system"]
                scores_dict[dataset][system] = _get_metrics(data["metrics"], dataset)

        ref_file = f"{input_dir}/{dataset}-references-scores.json"
        data = json.load(open(ref_file, "r"))
        scores_dict[dataset]["reference"] = _get_metrics(data["metrics"], dataset)

    return scores_dict


def _load_questeval_scores(input_dir: str):
    scores_dict = {}
    for dataset_dir in glob(f"{input_dir}/*"):
        dataset = os.path.basename(dataset_dir)
        data = json.load(open(f"{dataset_dir}/scores.json", "r"))
        scores_dict[dataset] = _get_metrics(data["metrics"], dataset)
    return scores_dict


def _load_rerank_scores(input_dir: str, metric: str):
    scores_dict = {}
    for dataset in DATASETS:
        scores = json.load(
            open(f"{input_dir}/{dataset}/standard/16/{metric}/scores.json", "r")
        )
        scores_dict[dataset] = _get_metrics(scores["metrics"], dataset)
    return scores_dict


def _rescale_values(x: List[float]) -> List[float]:
    min_x = min(x)
    max_x = max(x)
    return [(value - min_x) / (max_x - min_x) for value in x]


def _plot(
    submission_scores: Dict,
    opt_scores: Dict,
    ref_metric: str,
    ref_free_metric: str,
    opt_method: str,
    output_file: str,
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    for i, (dataset, ax1) in enumerate(zip(DATASETS, axes)):
        ax2 = ax1.twinx()

        x_values = []
        y_values = []

        x_values.append(opt_scores[dataset][ref_metric])
        y_values.append(opt_scores[dataset][ref_free_metric])

        for system, scores in submission_scores[dataset].items():
            if system == "reference":
                continue
            x_values.append(scores[ref_metric])
            y_values.append(scores[ref_free_metric])

        y_ref_value = submission_scores[dataset]["reference"][ref_free_metric]

        x_rescaled = _rescale_values(x_values)
        y_rescaled_with_ref = _rescale_values(y_values + [y_ref_value])
        y_rescaled = y_rescaled_with_ref[:-1]
        y_ref_rescaled = y_rescaled_with_ref[-1]

        for index, (x, y, x_rescale, y_rescale) in enumerate(
            zip(x_values, y_values, x_rescaled, y_rescaled)
        ):
            print(x, y, x_rescale, y_rescale)

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
        ax1.title.set_text(NAMES[dataset])

        if i == 0:
            if ref_metric == "rouge":
                ax1.set_ylabel("ROUGE-2")
            elif ref_metric == "bertscore":
                ax1.set_ylabel("BERTScore")
            elif ref_metric == "qaeval":
                ax1.set_ylabel("QAEval")
            else:
                raise Exception()
        else:
            if ref_free_metric == "questeval":
                ax2.set_ylabel("QuestEval")
            elif ref_free_metric == "blanc":
                ax2.set_ylabel("BLANC-Help")

    # Add a legend
    # https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    if opt_method == "questeval":
        opt_name = "QuestEval as a Model"
    elif opt_method == "questeval-rerank":
        opt_name = "QuestEval as a Model"
    elif opt_method == "blanc-rerank":
        opt_name = "BLANC as a Model"
    else:
        raise Exception()

    elements = [
        Line2D([0], [0], color="tab:blue", linewidth=3, label=opt_name),
        Line2D([0], [0], color="0.8", linewidth=3, label="Systems in Dataset"),
        Line2D(
            [0],
            [0],
            color="tab:red",
            marker="x",
            markeredgewidth=3,
            markersize=10,
            linestyle="None",
            label="Reference Summary",
        ),
    ]
    # fig.legend(handles=elements, ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.legend(handles=elements, ncol=2, loc="lower center")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(output_file)


def main(args):
    plt.rcParams.update({'font.size': 14})

    submission_scores = _load_submission_scores(args.submissions_dir)

    if args.opt_method == "questeval":
        questeval_scores = _load_questeval_scores(args.opt_method_dir)
        _plot(submission_scores, questeval_scores, args.ref_based_metric, args.ref_free_metric, args.opt_method, f"{args.output_dir}/questeval.pdf",)

    elif args.opt_method == "questeval-rerank":
        questeval_scores = _load_rerank_scores(args.opt_method_dir, args.ref_free_metric)
        _plot(submission_scores, questeval_scores, args.ref_based_metric, args.ref_free_metric, args.opt_method,
              f"{args.output_dir}/questeval-rerank.pdf")

    elif args.opt_method == "blanc-rerank":
        blanc_scores = _load_rerank_scores(args.opt_method_dir, args.ref_free_metric)
        _plot(submission_scores, blanc_scores, args.ref_based_metric, args.ref_free_metric, args.opt_method,
              f"{args.output_dir}/blanc-rerank.pdf")

    else:
        raise Exception()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--submissions-dir", required=True)
    argp.add_argument("--opt-method", required=True)
    argp.add_argument("--opt-method-dir", required=True)
    argp.add_argument("--ref-based-metric", required=True)
    argp.add_argument("--ref-free-metric", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)