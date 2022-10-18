import argparse
import json
import os
from repro.models.rei2020 import COMET
from repro.models.sellam2020 import BLEURT
from repro.models.thompson2020 import Prism
from repro.models.zhang2020 import BERTScore
from subprocess import check_output


def main(args):
    inputs = []
    inputs_src = []
    inputs_ref = []
    with open(args.candidate_file, "r") as f:
        candidates = f.read().splitlines()
        for candidate in candidates:
            inputs.append({"candidate": candidate})
            inputs_src.append({"candidate": candidate})
            inputs_ref.append({"candidate": candidate})

    if args.source_file is not None:
        with open(args.source_file, "r") as f:
            sources = f.read().splitlines()
            for inp1, inp2, source in zip(inputs, inputs_src, sources):
                inp1["sources"] = [source]
                inp2["sources"] = [source]

    if args.reference_file is not None:
        with open(args.reference_file, "r") as f:
            references = f.read().splitlines()
            for inp1, inp2, reference in zip(inputs, inputs_ref, references):
                inp1["references"] = [reference]
                inp2["references"] = [reference]

    metrics = {}

    if args.bleu:
        assert args.reference_file is not None
        stdout = check_output(
            f"sacrebleu {args.reference_file} -l {args.lp} -i {args.candidate_file} -m bleu -lc -tok intl -b -w 4",
            shell=True,
        )
        bleu = float(stdout.decode().strip())
        metrics["bleu"] = bleu

    if args.bleurt:
        assert args.reference_file is not None
        metric = BLEURT(device=args.device)
        bleurt, _ = metric.predict_batch(inputs_ref)
        metrics["bleurt"] = bleurt["bleurt"]["mean"]

    if args.comet:
        assert args.source_file is not None
        assert args.reference_file is not None
        metric = COMET(device=args.device)
        comet, _ = metric.predict_batch(inputs)
        metrics["comet"] = comet["comet"]

    if args.comet_src:
        assert args.source_file is not None
        metric = COMET(device=args.device)
        comet, _ = metric.predict_batch(inputs_src)
        metrics["comet-src"] = comet["comet-src"]

    if args.prism and "gu" not in args.lp:
        assert args.reference_file is not None
        target = args.lp.split("-")[1]
        metric = Prism(device=args.device, language=target)
        prism, _ = metric.predict_batch(inputs_ref)
        metrics["prism"] = prism["prism"]

    if args.prism_src and not args.lp.endswith("gu"):
        assert args.source_file is not None
        target = args.lp.split("-")[1]
        metric = Prism(device=args.device, language=target)
        prism, _ = metric.predict_batch(inputs_src)
        metrics["prism-src"] = prism["prism"]

    if args.bertscore:
        assert args.reference_file is not None
        target = args.lp.split("-")[1]
        metric = BERTScore(device=args.device, language=target)
        macro, _ = metric.predict_batch(inputs_ref)
        metrics["bertscore"] = macro["bertscore"]

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        out.write(json.dumps({"system": args.system_name, "metrics": metrics}) + "\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--candidate-file", required=True)
    argp.add_argument("--output-file", required=True)
    argp.add_argument("--system-name", required=True)
    argp.add_argument("--lp", required=True)
    argp.add_argument("--source-file")
    argp.add_argument("--reference-file")
    argp.add_argument("--device", type=int, required=True)
    argp.add_argument("--bleu", action="store_true")
    argp.add_argument("--bleurt", action="store_true")
    argp.add_argument("--comet", action="store_true")
    argp.add_argument("--comet-src", action="store_true")
    argp.add_argument("--prism", action="store_true")
    argp.add_argument("--prism-src", action="store_true")
    argp.add_argument("--bertscore", action="store_true")
    args = argp.parse_args()
    main(args)
