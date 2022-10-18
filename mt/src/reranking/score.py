import argparse
import math
import json
from joblib import Parallel, delayed
from repro.models.thompson2020 import Prism
from repro.models.rei2020 import COMET
from typing import List


def _score(instances: List, device: int, language: str) -> List:
    print(f"Scoring {len(instances)} instances on device {device}")

    # Convert into inputs for the metrics
    inputs = []
    for instance in instances:
        source = instance["source"]
        for prediction in instance["predictions"]:
            inputs.append({"sources": [source], "candidate": prediction["prediction"]})

    # Score with the metrics
    prism = Prism(device=device, language=language)
    _, prism_scores = prism.predict_batch(inputs)

    comet = COMET(device=device)
    _, comet_scores = comet.predict_batch(inputs)

    # Put the results into the prediction dicts
    index = 0
    for instance in instances:
        for prediction in instance["predictions"]:
            prediction["prism-src"] = prism_scores[index]
            prediction["comet-src"] = comet_scores[index]
            index += 1

    return instances


def main(args):
    instances = []
    with open(args.input_file, "r") as f_inp:
        with open(args.pred_file, "r") as f_pred:
            for line_inp, line_pred in zip(f_inp, f_pred):
                source = line_inp.strip()
                predictions = json.loads(line_pred)
                instances.append({"source": source, "predictions": predictions})

    # Divide into batches
    num_batches = len(args.devices)
    instances_per_batch = int(math.ceil(len(instances) / num_batches))
    batches = []
    for i in range(0, len(instances), instances_per_batch):
        batches.append(instances[i : i + instances_per_batch])

    # Distribute to the workers
    instances_list = Parallel(n_jobs=len(args.devices))(
        delayed(_score)(batch, device, args.language)
        for batch, device in zip(batches, args.devices)
    )

    with open(args.output_file, "w") as out:
        for instances in instances_list:
            for instance in instances:
                out.write(json.dumps(instance["predictions"]) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--pred-file", required=True)
    argp.add_argument("--devices", required=True, type=int, nargs="+")
    argp.add_argument("--language", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
