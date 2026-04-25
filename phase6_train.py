from __future__ import annotations

import argparse
import json
import os
from glob import glob

from app.evaluation import evaluate_policy, save_evaluation, save_json
from app.heuristic_policy import HeuristicPolicy
from app.model_policy import HuggingFaceModelPolicy
from app.plotting import write_line_chart_svg
from app.teacher_data import build_teacher_training_bundle, write_teacher_jsonl


def run_phase6_training(
    base_model: str,
    output_dir: str,
    variants_per_task: int,
    min_teacher_score: float,
    max_teacher_penalties: int,
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
) -> dict:
    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, LoraConfig, TaskType = _training_imports()

    os.makedirs(output_dir, exist_ok=True)
    bundle = build_teacher_training_bundle(
        variants_per_task=variants_per_task,
        min_score=min_teacher_score,
        max_penalties=max_teacher_penalties,
    )
    train_examples = bundle["train_examples"]
    eval_tasks = bundle["eval_tasks"]
    teacher_summary = bundle["summary"]

    if not train_examples:
        raise RuntimeError("No teacher-approved training examples were generated.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared_train_examples = _prepare_chat_training_examples(train_examples, tokenizer)
    dataset = Dataset.from_list(prepared_train_examples)
    dataset_split = dataset.train_test_split(test_size=max(1, min(4, len(prepared_train_examples) // 5)), seed=seed)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    dataset_dir = os.path.join(output_dir, "data")
    os.makedirs(dataset_dir, exist_ok=True)
    write_teacher_jsonl(os.path.join(dataset_dir, "teacher_train.jsonl"), prepared_train_examples)
    save_json(os.path.join(dataset_dir, "teacher_train_records.json"), bundle["train_records"])
    save_json(os.path.join(dataset_dir, "teacher_eval_records.json"), bundle["eval_records"])
    save_json(os.path.join(dataset_dir, "teacher_summary.json"), teacher_summary)

    baseline_policy = HuggingFaceModelPolicy(base_model=base_model, max_new_tokens=max_new_tokens)
    baseline_eval = evaluate_policy(baseline_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "baseline_eval.json"), baseline_eval)
    heuristic_eval = evaluate_policy(HeuristicPolicy(), eval_tasks)
    save_evaluation(os.path.join(output_dir, "heuristic_eval.json"), heuristic_eval)
    del baseline_policy

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        seed=seed,
        dataset_text_field="training_text",
        dataset_kwargs={"add_special_tokens": False},
        max_length=max_length,
        gradient_checkpointing=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        bf16=False,
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    trained_policy = HuggingFaceModelPolicy(
        base_model=base_model,
        adapter_path=output_dir,
        max_new_tokens=max_new_tokens,
    )
    trained_eval = evaluate_policy(trained_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "trained_eval.json"), trained_eval)

    reward_curve = _build_reward_curve(base_model, output_dir, eval_tasks, max_new_tokens)
    loss_curve = _extract_loss_curve(trainer.state.log_history)
    write_line_chart_svg(os.path.join(output_dir, "reward_curve.svg"), reward_curve, "Evaluation Reward By Epoch", "Avg Reward")
    write_line_chart_svg(os.path.join(output_dir, "loss_curve.svg"), loss_curve, "Training Loss", "Loss")

    with open(os.path.join(output_dir, "trainer_log_history.json"), "w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)

    summary = {
        "output_dir": output_dir,
        "base_model": base_model,
        "adapter_path": output_dir,
        "teacher_summary": teacher_summary,
        "train_case_count": len(prepared_train_examples),
        "eval_case_count": len(eval_tasks),
        "baseline_metrics": baseline_eval["metrics"],
        "trained_metrics": trained_eval["metrics"],
        "heuristic_metrics": heuristic_eval["metrics"],
        "improvement": _metric_delta(baseline_eval["metrics"], trained_eval["metrics"]),
        "gap_to_teacher": _metric_delta(trained_eval["metrics"], heuristic_eval["metrics"]),
    }
    save_json(os.path.join(output_dir, "training_summary.json"), summary)
    return summary


def _prepare_chat_training_examples(examples: list[dict], tokenizer) -> list[dict]:
    prepared: list[dict] = []
    for example in examples:
        training_text = example["text"]
        if getattr(tokenizer, "chat_template", None):
            training_text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        updated = dict(example)
        updated["training_text"] = training_text
        prepared.append(updated)
    return prepared


def _build_reward_curve(base_model: str, output_dir: str, eval_tasks: list[dict], max_new_tokens: int) -> list[float]:
    checkpoint_dirs = sorted(
        [path for path in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(path)],
        key=lambda path: int(path.rsplit("-", 1)[-1]),
    )
    if not checkpoint_dirs:
        return []

    rewards: list[float] = []
    for checkpoint_dir in checkpoint_dirs:
        policy = HuggingFaceModelPolicy(
            base_model=base_model,
            adapter_path=checkpoint_dir,
            max_new_tokens=max_new_tokens,
            allow_heuristic_fallback=False,
        )
        evaluation = evaluate_policy(policy, eval_tasks)
        rewards.append(float(evaluation["metrics"]["avg_reward"]))
    return rewards


def _extract_loss_curve(log_history: list[dict]) -> list[float]:
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    if losses:
        return losses
    return [float(item["eval_loss"]) for item in log_history if "eval_loss" in item]


def _metric_delta(left: dict, right: dict) -> dict:
    tracked = [
        "avg_reward",
        "avg_score",
        "intent_accuracy",
        "risk_accuracy",
        "action_accuracy",
        "callback_correctness",
        "priority_pain_recall",
        "unsafe_action_rate",
        "summary_completeness",
    ]
    return {key: float(right.get(key, 0.0)) - float(left.get(key, 0.0)) for key in tracked}


def _training_imports():
    try:
        from pathlib import Path

        original_read_text = Path.read_text

        def read_text_utf8(self, encoding=None, errors=None):
            return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

        Path.read_text = read_text_utf8

        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "TRL training dependencies are missing. Install torch, transformers, datasets, peft, accelerate, and trl."
        ) from exc

    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, LoraConfig, TaskType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 teacher-distillation training for PhysioSupportEnv.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune.")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase6/training", help="Directory for checkpoints and reports.")
    parser.add_argument("--variants-per-task", type=int, default=8, help="Synthetic variants created per base task.")
    parser.add_argument("--min-teacher-score", type=float, default=0.8, help="Minimum teacher score required for a training example.")
    parser.add_argument("--max-teacher-penalties", type=int, default=0, help="Maximum penalty count allowed for teacher examples.")
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum tokenized sequence length.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length for policy evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_phase6_training(
        base_model=args.base_model,
        output_dir=args.output_dir,
        variants_per_task=args.variants_per_task,
        min_teacher_score=args.min_teacher_score,
        max_teacher_penalties=args.max_teacher_penalties,
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
