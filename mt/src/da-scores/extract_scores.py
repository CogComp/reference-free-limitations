import argparse
import gzip
import json
import os
from collections import defaultdict
from glob import glob


def main(args):
    # scores_dict[lp][system][metric]
    scores_dict = defaultdict(lambda: defaultdict(dict))

    # Load the DA scores
    da_file = f"{args.wmt19_dir}/manual-evaluation/DA-syslevel.csv"
    with open(da_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                # Header
                continue
            lp, score, system = line.split()
            scores_dict[lp][system]["DA"] = float(score)

    # Load the baselines
    baseline_dir = f"{args.wmt19_dir}/baselines"
    for gz in glob(f"{baseline_dir}/*.sys.score.gz"):
        with gzip.open(gz, "rb") as f:
            for line in f:
                metric, lp, _, system, score = line.decode().strip().split()
                scores_dict[lp][system][metric] = float(score)

    # Load the metric submissions
    submission_dir = f"{args.wmt19_dir}/final-metric-scores/submissions-corrected"
    for gz in glob(f"{submission_dir}/*.sys.score.gz"):
        with gzip.open(gz, "rb") as f:
            for line in f:
                metric, lp, _, system, score, _, _ = line.decode().strip().split()
                scores_dict[lp][system][metric] = float(score)

    # Save the data
    os.makedirs(args.output_dir)
    for lp in scores_dict.keys():
        with open(f"{args.output_dir}/{lp}.jsonl", "w") as out:
            for system, metrics in scores_dict[lp].items():
                out.write(json.dumps({"system": system, "metrics": metrics}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--wmt19-dir", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)
