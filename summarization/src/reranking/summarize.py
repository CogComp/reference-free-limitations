import argparse
import json
from repro.models.lewis2020 import BART


def main(args):
    model = BART(
        nbest=args.beam_size,
        beam_size=args.beam_size,
        device=args.device
    )

    # Keep track of the instance_ids that we've seen
    seen = set()
    instances = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            instance_id = data["instance_id"]
            if instance_id in seen:
                continue
            seen.add(instance_id)

            if "document" in data:
                source = data["document"]["text"]
            else:
                source = data["documents"][0]["text"]

            instances.append({
                "instance_id": instance_id,
                "document": source,
            })

    summaries_list = model.predict_batch(instances)

    with open(args.output_file, "w") as out:
        for instance, summaries in zip(instances, summaries_list):
            instance_id = instance["instance_id"]
            if args.beam_size == 1:
                summaries = [summaries]

            out.write(json.dumps({
                "instance_id": instance_id,
                "predictions": [
                    {"prediction": summary} for summary in summaries
                ]
            }) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--beam-size", required=True, type=int)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)