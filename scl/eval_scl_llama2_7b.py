import argparse
import json
from lm_eval import evaluator

TASKS = [
    #"wikitext",
    "c4",
    #"winogrande",
    #"rte",
    #"piqa",
    #"arc_challenge",
]

def run_eval(model_dir, task, device="cuda"):
    print(f"\n====== Running task: {task} ======\n")

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_dir},dtype=float16",
        tasks=[task],
        batch_size=8,
        device=device,
        limit=1000
    )

    print(json.dumps(results["results"], indent=2))
    return results["results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    summary = {}

    print("\n=== Evaluating model:", args.model_dir, "===\n")

    for task in TASKS:
        result = run_eval(args.model_dir, task, args.device)
        summary[task] = result

    print("\n=========== FINAL SUMMARY ===========\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
