import argparse
import json
import numpy as np
import os


def main(args):
    questeval_before = []
    questeval_after = []
    blanc_before = []
    blanc_after = []

    os.makedirs(os.path.dirname(args.standard_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.questeval_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.blanc_file), exist_ok=True)

    with open(args.standard_file, "w") as out_standard:
        with open(args.questeval_file, "w") as out_questeval:
            with open(args.blanc_file, "w") as out_blanc:
                with open(args.score_file, "r") as f:
                    for line in f:
                        instance = json.loads(line)
                        instance_id = instance["instance_id"]

                        predictions = instance["predictions"]
                        out_standard.write(json.dumps({
                            "instance_id": instance_id,
                            "summarizer_id": "standard-opt",
                            "summary": predictions[0]["prediction"]
                        }) + "\n")
                        questeval_before.append(predictions[0]["questeval"])
                        blanc_before.append(predictions[0]["blanc"])

                        predictions.sort(
                            key=lambda pred: pred["questeval"], reverse=True
                        )
                        out_questeval.write(json.dumps({
                            "instance_id": instance_id,
                            "summarizer_id": "questeval-opt",
                            "summary": predictions[0]["prediction"]
                        }) + "\n")
                        questeval_after.append(predictions[0]["questeval"])

                        predictions.sort(
                            key=lambda pred: pred["blanc"],
                            reverse=True,
                        )
                        out_blanc.write(json.dumps({
                            "instance_id": instance_id,
                            "summarizer_id": "blanc-opt",
                            "summary": predictions[0]["prediction"]
                        }) + "\n")
                        blanc_after.append(predictions[0]["blanc"])

    questeval_before = np.mean(questeval_before) * 100
    questeval_after = np.mean(questeval_after) * 100
    questeval_improvement = (questeval_after - questeval_before) / questeval_before * 100

    blanc_before = np.mean(blanc_before)
    blanc_after = np.mean(blanc_after)
    blanc_improvement = (blanc_after - blanc_before) / blanc_before * 100

    print(
        f"QuestEval: {np.mean(questeval_before):.4f} -> {np.mean(questeval_after):.4f} (+{questeval_improvement:.2f}%)"
    )
    print(
        f"BLANC: {np.mean(blanc_before):.4f} -> {np.mean(blanc_after):.4f} (+{blanc_improvement:.2f}%)"
    )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--score-file", required=True)
    argp.add_argument("--standard-file", required=True)
    argp.add_argument("--questeval-file", required=True)
    argp.add_argument("--blanc-file", required=True)
    args = argp.parse_args()
    main(args)
