import argparse
import json
import torch
from tqdm import tqdm


def main(args):
    if args.lp in ["en-de", "de-en", "en-ru", "ru-en"]:
        model = torch.hub.load(
            "pytorch/fairseq",
            f"transformer.wmt19.{args.lp}",
            tokenizer="moses",
            bpe="fastbpe",
            checkpoint_file="model1.pt:model2.pt:model3.pt:model4.pt",
        )
    else:
        raise Exception(f"Unknown lp: {args.lp}")
    model.eval()
    model.cuda()

    with open(args.input_file, "r") as f:
        sources = f.read().splitlines()

    with open(args.output_file, "w") as out:
        for source in tqdm(sources):
            source_bin = model.encode(source)

            if args.inference_method == "standard":
                predictions = model.generate(source_bin, beam=args.beam_size)
            elif args.inference_method == "diverse":
                assert args.diverse_beam_groups > 0
                predictions = model.generate(
                    source_bin,
                    beam=args.beam_size,
                    diverse_beam_groups=args.diverse_beam_groups,
                )
            elif args.inference_method == "sampling":
                predictions = model.generate(
                    source_bin,
                    beam=args.beam_size,
                    sampling=True,
                    sampling_topk=args.sampling_topk,
                )
            else:
                raise Exception(f"Unknown inference method: {args.inference_method}")

            hypotheses = [
                {"prediction": model.decode(pred["tokens"])} for pred in predictions
            ]
            out.write(json.dumps(hypotheses) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--lp", required=True, choices=["en-de", "de-en", "en-ru", "ru-en"]
    )
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--beam-size", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    argp.add_argument(
        "--inference-method", required=True, choices=["standard", "diverse", "sampling"]
    )
    # Diverse
    argp.add_argument("--diverse-beam-groups", type=int, default=-1)
    # Sampling
    argp.add_argument("--sampling-topk", type=int, default=-1)
    args = argp.parse_args()
    main(args)
