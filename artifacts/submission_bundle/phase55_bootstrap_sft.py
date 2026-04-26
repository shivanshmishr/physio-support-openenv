from __future__ import annotations

import argparse
import json
import os

from app.evaluation import save_json
from warmup_sft import run_warmup


def run_phase55_bootstrap_sft(
    base_model: str,
    output_dir: str,
    variants_per_task: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    max_new_tokens: int,
    showcase_limit: int,
    training_data_mode: str,
    teacher_min_score: float,
    teacher_max_penalties: int,
) -> dict:
    summary = run_warmup(
        base_model=base_model,
        output_dir=output_dir,
        variants_per_task=variants_per_task,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        seed=seed,
        max_new_tokens=max_new_tokens,
        showcase_limit=showcase_limit,
        training_data_mode=training_data_mode,
        teacher_min_score=teacher_min_score,
        teacher_max_penalties=teacher_max_penalties,
    )
    summary = {
        "training_type": "bootstrap_sft",
        "bootstrap_source": {
            "teacher": "heuristic_teacher_sft",
            "structured": "structured_target_sft",
            "hybrid": "hybrid_teacher_plus_structured_sft",
        }[training_data_mode],
        **summary,
    }
    save_json(os.path.join(output_dir, "bootstrap_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5.5 bootstrap SFT to produce a PEFT adapter for downstream Phase 6 GRPO."
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/phase55/bootstrap_sft",
        help="Directory for the bootstrap adapter and reports.",
    )
    parser.add_argument("--variants-per-task", type=int, default=8, help="Synthetic variants created per base task.")
    parser.add_argument("--num-train-epochs", type=float, default=2.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum tokenized sequence length.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length for bootstrap evaluation.")
    parser.add_argument("--showcase-limit", type=int, default=3, help="How many representative examples to save.")
    parser.add_argument(
        "--training-data-mode",
        type=str,
        choices=["structured", "teacher", "hybrid"],
        default="hybrid",
        help="Bootstrap SFT source. Hybrid mode combines structured truth targets with high-scoring heuristic demonstrations.",
    )
    parser.add_argument(
        "--teacher-min-score",
        type=float,
        default=0.8,
        help="Minimum teacher task score required when training-data-mode=teacher.",
    )
    parser.add_argument(
        "--teacher-max-penalties",
        type=int,
        default=0,
        help="Maximum number of teacher penalties allowed when training-data-mode=teacher.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_phase55_bootstrap_sft(
        base_model=args.base_model,
        output_dir=args.output_dir,
        variants_per_task=args.variants_per_task,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        showcase_limit=args.showcase_limit,
        training_data_mode=args.training_data_mode,
        teacher_min_score=args.teacher_min_score,
        teacher_max_penalties=args.teacher_max_penalties,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
