import argparse
import json
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from questeval.questeval_metric import QuestEval


def main(args):
    num_tokens = 512
    questeval = QuestEval(task="summarization", do_weighter=True, isCuda=True)

    # Keeps track of which instances we've already processed
    seen = set()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as out:
        with open(args.input_file, "r") as f:
            for line in tqdm(f):
                instance = json.loads(line)
                instance_id = instance["instance_id"]
                if instance_id in seen:
                    continue
                seen.add(instance_id)

                if "document" in instance:
                    document = instance["document"]["text"]
                else:
                    document = instance["documents"][0]["text"]
                if isinstance(document, list):
                    document = " ".join(document)

                # Take the first `num_tokens` tokens
                document = " ".join(document.split()[:num_tokens])

                # Sentence split
                sentences = sent_tokenize(document)

                if len(sentences) <= args.num_sents:
                    summary = sentences
                else:
                    best_indices = []
                    candidate_indices = list(range(len(sentences)))
                    global_best = None
                    for _ in range(args.num_sents):
                        candidate_summaries = []
                        for index in candidate_indices:
                            candidate_summaries.append(" ".join([sentences[i] for i in sorted(best_indices + [index])]))

                        scores = []
                        for candidate in candidate_summaries:
                            score_dict = questeval.compute_all(
                                hypothesis=candidate,
                                source=document
                            )
                            scores.append(score_dict["scores"]["fscore"])

                        best_index = None
                        best_score = None
                        for index, score in zip(candidate_indices, scores):
                            if best_score is None or score > best_score:
                                best_score = score
                                best_index = index

                        if global_best is None or best_score > global_best:
                            global_best = best_score
                            best_indices.append(best_index)
                            candidate_indices.remove(best_index)
                            tqdm.write(str(best_score))
                        else:
                            # Adding this sentence makes the score worse. Terminate early
                            break

                    tqdm.write(str(best_indices))
                    summary = [sentences[index] for index in sorted(best_indices)]

                out.write(json.dumps({
                    "instance_id": instance_id,
                    "summarizer_id": "questeval-ext-opt",
                    "summary": " ".join(summary)
                }) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--num-sents", required=True, type=int)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)