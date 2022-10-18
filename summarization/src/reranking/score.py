import argparse
import json
from questeval.questeval_metric import QuestEval
from repro.models.vasilyev2020 import BLANCHelp
from tqdm import tqdm
from typing import Dict


def _load_sources(input_file: str) -> Dict[str, str]:
    sources = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            instance_id = data["instance_id"]
            if instance_id in sources:
                continue
            if "document" in data:
                sources[instance_id] = data["document"]["text"]
            else:
                sources[instance_id] = data["documents"][0]["text"]
    return sources


def main(args):
    num_tokens = 512
    questeval = QuestEval(task="summarization", do_weighter=True, isCuda=True)

    sources = _load_sources(args.input_file)
    instances = []
    inputs = []
    with open(args.pred_file, "r") as f:
        for line in tqdm(f):
            instance = json.loads(line)
            instances.append(instance)

            instance_id = instance["instance_id"]
            source = sources[instance_id]
            truncated_source = " ".join(source.split()[:num_tokens])

            for pred_dict in instance["predictions"]:
                prediction = pred_dict["prediction"]
                score_dict = questeval.compute_all(
                    hypothesis=prediction,
                    source=truncated_source
                )
                pred_dict["questeval"] = score_dict["scores"]["fscore"]

                inputs.append({"sources": [source], "candidate": pred_dict["prediction"]})

    blanc = BLANCHelp(device=args.device)
    _, blanc_scores = blanc.predict_batch(inputs)

    index = 0
    for instance in instances:
        for prediction in instance["predictions"]:
            prediction["blanc"] = blanc_scores[index]["blanc-help"]
            index += 1

    with open(args.output_file, "w") as out:
        for instance in instances:
            out.write(json.dumps(instance) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--pred-file", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)