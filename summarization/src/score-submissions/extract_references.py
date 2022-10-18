import argparse
import json


def main(args):
    seen = set()
    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                if instance_id in seen:
                    continue
                seen.add(instance_id)

                if "document" in data:
                    document = data["document"]
                else:
                    document = data["documents"][0]

                if "reference" in data:
                    reference = data["reference"]["text"]
                else:
                    reference = data["references"][0]["text"]

                out.write(json.dumps({
                    "instance_id": instance_id,
                    "summarizer_id": "reference",
                    "document": document,
                    "summary": {"text": reference},
                    "reference": {"text": reference}
                }) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)