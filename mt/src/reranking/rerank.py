import argparse
import json
import numpy as np
import os
from typing import List


def main(args):
    prism_before = []
    prism_after = []
    comet_before = []
    comet_after = []

    os.makedirs(os.path.dirname(args.standard_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.prism_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.comet_file), exist_ok=True)

    with open(args.standard_file, "w") as out_standard:
        with open(args.prism_file, "w") as out_prism:
            with open(args.comet_file, "w") as out_comet:
                with open(args.score_file, "r") as f:
                    for line in f:
                        predictions: List = json.loads(line)
                        out_standard.write(predictions[0]["prediction"] + "\n")
                        prism_before.append(predictions[0]["prism-src"]["prism"])
                        comet_before.append(predictions[0]["comet-src"]["comet-src"])

                        predictions.sort(
                            key=lambda pred: pred["prism-src"]["prism"], reverse=True
                        )
                        out_prism.write(predictions[0]["prediction"] + "\n")
                        prism_after.append(predictions[0]["prism-src"]["prism"])

                        predictions.sort(
                            key=lambda pred: pred["comet-src"]["comet-src"],
                            reverse=True,
                        )
                        out_comet.write(predictions[0]["prediction"] + "\n")
                        comet_after.append(predictions[0]["comet-src"]["comet-src"])

    prism_before = np.mean(prism_before)
    prism_after = np.mean(prism_after)
    prism_improvement = (prism_after - prism_before) / prism_before * 100

    comet_before = np.mean(comet_before)
    comet_after = np.mean(comet_after)
    comet_improvement = (comet_after - comet_before) / comet_before * 100

    print(
        f"Prism-src: {np.mean(prism_before):.4f} -> {np.mean(prism_after):.4f} (+{prism_improvement:.2f}%)"
    )
    print(
        f"Comet-src: {np.mean(comet_before):.4f} -> {np.mean(comet_after):.4f} (+{comet_improvement:.2f}%)"
    )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--score-file", required=True)
    argp.add_argument("--standard-file", required=True)
    argp.add_argument("--prism-file", required=True)
    argp.add_argument("--comet-file", required=True)
    args = argp.parse_args()
    main(args)
