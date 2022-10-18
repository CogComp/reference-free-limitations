import argparse
import json
import numpy as np
import os
from collections import defaultdict
from questeval.questeval_metric import QuestEval
from repro.models.deutsch2021 import QAEval
from repro.models.lin2004 import ROUGE
from repro.models.vasilyev2020 import BLANCHelp
from repro.models.zhang2020 import BERTScore
from typing import Dict, List, Tuple


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


def _load_references(input_file: str) -> Dict[str, str]:
    references = {}
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            instance_id = data["instance_id"]
            if instance_id in references:
                continue
            references[instance_id] = data["references"][0]["text"]
    return references


def _load_candidates(input_file: str) -> Dict[str, Dict[str, str]]:
    candidates = defaultdict(dict)
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            system = data["summarizer_id"]
            instance_id = data["instance_id"]

            if "summary" in data:
                if isinstance(data["summary"], dict) and "text" in data["summary"]:
                    summary = data["summary"]["text"]
                else:
                    summary = data["summary"]

                if isinstance(summary, list):
                    summary = " ".join(summary)
                candidates[system][instance_id] = summary
            else:
                raise Exception()
    return candidates


def _convert_to_inputs(
    candidates: Dict[str, str],
    references: Dict[str, str],
    sources: Dict[str, str],
) -> Tuple[List, List, List]:
    inputs = []
    inputs_ref = []
    inputs_src = []

    for instance_id, candidate in candidates.items():
        if references is not None:
            inputs_ref.append({
                "candidate": candidate,
                "references": [references[instance_id]]
            })

        if sources is not None:
            inputs_src.append({
                "candidate": candidate,
                "sources": [sources[instance_id]]
            })

        if references is not None and sources is not None:
            inputs.append({
                "candidate": candidate,
                "references": [references[instance_id]],
                "sources": [sources[instance_id]]
            })

    return inputs, inputs_ref, inputs_src


def main(args):
    sources = None
    if args.source_file:
        sources = _load_sources(args.source_file)

    references = None
    if args.reference_file:
        references = _load_references(args.reference_file)

    candidates_dict = _load_candidates(args.candidate_file)

    metrics = defaultdict(dict)
    for system, candidates in candidates_dict.items():
        inputs, inputs_ref, inputs_src = _convert_to_inputs(
            candidates, references, sources
        )

        if args.rouge:
            assert references is not None
            metric = ROUGE()
            macro, _ = metric.predict_batch(inputs_ref)
            metrics[system]["rouge"] = macro

        if args.bertscore:
            assert references is not None
            metric = BERTScore(device=args.device)
            macro, _ = metric.predict_batch(inputs_ref)
            metrics[system]["bertscore"] = macro["bertscore"]

        if args.qaeval:
            assert references is not None
            metric = QAEval(device=args.device)
            macro, _ = metric.predict_batch(inputs_ref)
            metrics[system]["qaeval"] = macro["qa-eval"]

        if args.questeval:
            assert sources is not None
            num_tokens = 512
            questeval = QuestEval(task="summarization", do_weighter=True, isCuda=True)

            scores = []
            for inp in inputs_src:
                candidate = inp["candidate"]
                source = inp["sources"][0]
                truncated_source = " ".join(source.split()[:num_tokens])
                score_dict = questeval.compute_all(
                    hypothesis=candidate,
                    source=truncated_source
                )
                scores.append(score_dict["scores"]["fscore"])
            metrics[system]["questeval"] = np.mean(scores)
            del questeval

        if args.blanc:
            assert sources is not None
            metric = BLANCHelp(device=args.device)
            macro, _ = metric.predict_batch(inputs_src)
            metrics[system]["blanc"] = macro["blanc-help"]

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(args.output_file, "w") as out:
        for system, m in metrics.items():
            out.write(json.dumps({"system": system, "metrics": m}) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--candidate-file", required=True)
    argp.add_argument("--output-file", required=True)
    argp.add_argument("--device", type=int, required=True)
    argp.add_argument("--source-file")
    argp.add_argument("--reference-file")
    argp.add_argument("--rouge", action="store_true")
    argp.add_argument("--bertscore", action="store_true")
    argp.add_argument("--qaeval", action="store_true")
    argp.add_argument("--questeval", action="store_true")
    argp.add_argument("--blanc", action="store_true")
    args = argp.parse_args()
    main(args)