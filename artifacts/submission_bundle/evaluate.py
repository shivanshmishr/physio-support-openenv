from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from app.evaluation import build_showcase_examples, build_task_manifest, evaluate_policy, save_evaluation, save_json
from app.heuristic_policy import HeuristicPolicy
from app.model_policy import HuggingFaceModelPolicy
from app.training_data import build_sft_splits, build_split_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy against the PhysioSupportEnv reward engine.")
    parser.add_argument(
        "--policy",
        choices=["heuristic", "baseline", "trained"],
        default="heuristic",
        help="Which policy to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model used for baseline or trained policy evaluation.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="artifacts/training",
        help="Adapter path used when --policy trained.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval", "all"],
        default="eval",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--variants-per-task", type=int, default=8, help="Synthetic variants created per base task.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length for model policies.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--artifact-dir", type=str, default="", help="Optional directory for Phase 4 evaluation artifacts.")
    parser.add_argument("--showcase-limit", type=int, default=3, help="How many representative examples to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_examples, eval_examples = build_sft_splits(variants_per_task=args.variants_per_task)
    split_summary = build_split_summary(variants_per_task=args.variants_per_task)

    if args.split == "train":
        tasks = [dict(example["task"]) for example in train_examples]
    elif args.split == "eval":
        tasks = [dict(example["task"]) for example in eval_examples]
    else:
        tasks = [dict(example["task"]) for example in train_examples + eval_examples]

    if args.policy == "heuristic":
        policy = HeuristicPolicy()
    elif args.policy == "baseline":
        policy = HuggingFaceModelPolicy(base_model=args.base_model, max_new_tokens=args.max_new_tokens)
    else:
        if not os.path.exists(args.adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
        policy = HuggingFaceModelPolicy(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
        )

    evaluation = evaluate_policy(policy, tasks)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_evaluation(args.output, evaluation)

    if args.artifact_dir:
        write_phase4_artifacts(
            artifact_dir=args.artifact_dir,
            evaluation=evaluation,
            tasks=tasks,
            split_summary=split_summary,
            policy_name=args.policy,
            split_name=args.split,
            base_model=args.base_model,
            adapter_path=args.adapter_path if args.policy == "trained" else "",
            variants_per_task=args.variants_per_task,
            showcase_limit=args.showcase_limit,
        )

    print(json.dumps(evaluation["metrics"], indent=2))


def write_phase4_artifacts(
    artifact_dir: str,
    evaluation: dict,
    tasks: list[dict],
    split_summary: dict,
    policy_name: str,
    split_name: str,
    base_model: str,
    adapter_path: str,
    variants_per_task: int,
    showcase_limit: int,
) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    task_manifest = build_task_manifest(tasks)
    showcase_examples = build_showcase_examples(evaluation["results"], limit=showcase_limit)
    run_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy_name,
        "split": split_name,
        "variants_per_task": variants_per_task,
        "base_model": base_model if policy_name in {"baseline", "trained"} else "",
        "adapter_path": adapter_path if policy_name == "trained" else "",
        "num_tasks": len(tasks),
    }

    save_evaluation(os.path.join(artifact_dir, "evaluation.json"), evaluation)
    save_json(os.path.join(artifact_dir, "metrics.json"), evaluation["metrics"])
    save_json(os.path.join(artifact_dir, "task_manifest.json"), task_manifest)
    save_json(os.path.join(artifact_dir, "split_summary.json"), split_summary)
    save_json(os.path.join(artifact_dir, "showcase_examples.json"), showcase_examples)
    save_json(os.path.join(artifact_dir, "run_manifest.json"), run_manifest)

    summary_lines = [
        "# Phase 4 Baseline Evaluation",
        "",
        f"- policy: `{policy_name}`",
        f"- split: `{split_name}`",
        f"- variants_per_task: `{variants_per_task}`",
        f"- num_tasks: `{len(tasks)}`",
        "",
        "## Metrics",
    ]
    for key, value in evaluation["metrics"].items():
        if isinstance(value, dict):
            continue
        summary_lines.append(f"- {key}: `{value}`")
    summary_lines.extend(
        [
            "",
            "## Showcase Examples",
        ]
    )
    for example in showcase_examples:
        summary_lines.append(
            f"- `{example['task_id']}` ({example['selection_reason']}): "
            f"score=`{example['score']}`, reward=`{example['reward']}`, "
            f"action_exact=`{example['action_exact']}`, intent_exact=`{example['intent_exact']}`"
        )

    with open(os.path.join(artifact_dir, "phase4_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
