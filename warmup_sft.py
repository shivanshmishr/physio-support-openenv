from __future__ import annotations

import argparse
import json
import os
from glob import glob

from app.evaluation import save_evaluation, save_json
from app.heuristic_policy import HeuristicPolicy
from app.model_policy import HuggingFaceModelPolicy
from app.plotting import write_line_chart_svg
from app.training_data import build_sft_splits, build_split_summary
from app.warmup import build_warmup_showcase, evaluate_warmup_policy


def run_warmup(
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
) -> dict:
    torch, Dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, LoraConfig, TaskType = _training_imports()

    os.makedirs(output_dir, exist_ok=True)
    train_examples, eval_examples = build_sft_splits(variants_per_task=variants_per_task)
    split_summary = build_split_summary(variants_per_task=variants_per_task)
    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    dataset_dir = os.path.join(output_dir, "data")
    os.makedirs(dataset_dir, exist_ok=True)
    _write_jsonl(os.path.join(dataset_dir, "train.jsonl"), train_examples)
    _write_jsonl(os.path.join(dataset_dir, "eval.jsonl"), eval_examples)
    save_json(os.path.join(dataset_dir, "split_summary.json"), split_summary)

    eval_tasks = [dict(example["task"]) for example in eval_examples]

    baseline_policy = HuggingFaceModelPolicy(base_model=base_model, max_new_tokens=max_new_tokens)
    baseline_schema = evaluate_warmup_policy(baseline_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "baseline_schema_eval.json"), baseline_schema)
    heuristic_schema = evaluate_warmup_policy(HeuristicPolicy(), eval_tasks)
    save_evaluation(os.path.join(output_dir, "heuristic_schema_eval.json"), heuristic_schema)
    del baseline_policy

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

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
        dataset_text_field="text",
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
    trained_schema = evaluate_warmup_policy(trained_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "trained_schema_eval.json"), trained_schema)

    schema_curve = _build_schema_curve(base_model, output_dir, eval_tasks, max_new_tokens)
    loss_curve = _extract_loss_curve(trainer.state.log_history)
    write_line_chart_svg(
        os.path.join(output_dir, "schema_validity_curve.svg"),
        schema_curve,
        "Schema Validity By Epoch",
        "Schema Valid Rate",
    )
    write_line_chart_svg(os.path.join(output_dir, "warmup_loss_curve.svg"), loss_curve, "Warm-up Training Loss", "Loss")

    history_path = os.path.join(output_dir, "trainer_log_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)

    showcase = {
        "baseline": build_warmup_showcase(baseline_schema["results"], limit=showcase_limit),
        "trained": build_warmup_showcase(trained_schema["results"], limit=showcase_limit),
    }
    save_json(os.path.join(output_dir, "warmup_showcase.json"), showcase)

    summary = {
        "output_dir": output_dir,
        "base_model": base_model,
        "adapter_path": output_dir,
        "train_case_count": len(train_examples),
        "eval_case_count": len(eval_examples),
        "split_summary": split_summary,
        "baseline_schema_metrics": baseline_schema["metrics"],
        "trained_schema_metrics": trained_schema["metrics"],
        "heuristic_schema_metrics": heuristic_schema["metrics"],
        "schema_improvement": _metric_delta(baseline_schema["metrics"], trained_schema["metrics"]),
    }
    save_json(os.path.join(output_dir, "warmup_summary.json"), summary)
    return summary


def _build_schema_curve(base_model: str, output_dir: str, eval_tasks: list[dict], max_new_tokens: int) -> list[float]:
    checkpoint_dirs = sorted(
        [path for path in glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(path)],
        key=lambda path: int(path.rsplit("-", 1)[-1]),
    )
    if not checkpoint_dirs:
        return []

    schema_rates: list[float] = []
    for checkpoint_dir in checkpoint_dirs:
        policy = HuggingFaceModelPolicy(
            base_model=base_model,
            adapter_path=checkpoint_dir,
            max_new_tokens=max_new_tokens,
            allow_heuristic_fallback=False,
        )
        evaluation = evaluate_warmup_policy(policy, eval_tasks)
        schema_rates.append(float(evaluation["metrics"]["schema_valid_rate"]))
    return schema_rates


def _extract_loss_curve(log_history: list[dict]) -> list[float]:
    losses = [float(item["loss"]) for item in log_history if "loss" in item]
    if losses:
        return losses
    eval_losses = [float(item["eval_loss"]) for item in log_history if "eval_loss" in item]
    return eval_losses


def _metric_delta(baseline: dict, trained: dict) -> dict:
    tracked = [
        "schema_valid_rate",
        "required_keys_rate",
        "allowed_action_rate",
        "secondary_actions_rate",
        "avg_score_on_valid",
        "avg_reward_on_valid",
    ]
    return {key: float(trained.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in tracked}


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


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
            "TRL warm-up dependencies are missing. Install torch, transformers, datasets, peft, accelerate, and trl."
        ) from exc

    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, LoraConfig, TaskType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 warm-up SFT for schema-compliant structured outputs.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune.")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase5/warmup", help="Directory for checkpoints and reports.")
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
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generation length for schema evaluation.")
    parser.add_argument("--showcase-limit", type=int, default=3, help="How many representative examples to save.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_warmup(
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
