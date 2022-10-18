import argparse
import os
from repro.models.thompson2020 import Prism


def main(args):
    inputs = []
    with open(args.input_file, "r") as f:
        for line in f:
            inputs.append({"source": line.strip()})

    model = Prism(device=args.device)
    translations = model.translate_batch(args.language, inputs, batch_size=8)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as out:
        for translation in translations:
            out.write(translation + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--language", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
