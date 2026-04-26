from __future__ import annotations

import argparse
import inspect
import json
import os
from glob import glob

from app.evaluation import evaluate_policy, save_evaluation, save_json
from app.heuristic_policy import HeuristicPolicy
from app.model_policy import HuggingFaceModelPolicy
from app.plotting import write_line_chart_svg
from app.rl_training import EnvironmentRewardFunction, build_grpo_training_bundle, write_jsonl
from phase55_bootstrap_sft import run_phase55_bootstrap_sft


def run_phase6_training(
    base_model: str,
    output_dir: str,
    variants_per_task: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_prompt_length: int,
    max_completion_length: int,
    num_generations: int,
    temperature: float,
    top_p: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    reward_mode: str,
    summary_bonus_scale: float,
    reply_bonus_scale: float,
    exact_action_bonus_scale: float,
    acceptable_action_penalty_scale: float,
    bootstrap_adapter_path: str,
    bootstrap_auto: bool,
    bootstrap_output_dir: str,
    bootstrap_num_train_epochs: float,
    bootstrap_learning_rate: float,
    bootstrap_per_device_eval_batch_size: int,
    bootstrap_max_length: int,
    bootstrap_showcase_limit: int,
    bootstrap_training_data_mode: str,
    bootstrap_teacher_min_score: float,
    bootstrap_teacher_max_penalties: int,
) -> dict:
    (
        torch,
        Dataset,
        AutoModelForCausalLM,
        AutoTokenizer,
        GRPOConfig,
        GRPOTrainer,
        LoraConfig,
        TaskType,
        PeftModel,
    ) = _training_imports()

    os.makedirs(output_dir, exist_ok=True)
    bundle = build_grpo_training_bundle(variants_per_task=variants_per_task)
    train_rows = bundle["train_rows"]
    eval_tasks = bundle["eval_tasks"]
    split_summary = bundle["split_summary"]

    dataset_dir = os.path.join(output_dir, "data")
    os.makedirs(dataset_dir, exist_ok=True)
    write_jsonl(os.path.join(dataset_dir, "train_prompts.jsonl"), train_rows)
    save_json(os.path.join(dataset_dir, "split_summary.json"), split_summary)

    baseline_policy = HuggingFaceModelPolicy(
        base_model=base_model,
        max_new_tokens=max_completion_length,
        temperature=0.0,
    )
    baseline_eval = evaluate_policy(baseline_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "baseline_eval.json"), baseline_eval)
    heuristic_eval = evaluate_policy(HeuristicPolicy(), eval_tasks)
    save_evaluation(os.path.join(output_dir, "heuristic_eval.json"), heuristic_eval)
    del baseline_policy

    bootstrap_summary = None
    resolved_bootstrap_path = bootstrap_adapter_path.strip()
    if not resolved_bootstrap_path and bootstrap_auto:
        resolved_bootstrap_path, bootstrap_summary = _prepare_bootstrap_adapter(
            base_model=base_model,
            output_dir=bootstrap_output_dir or os.path.join(output_dir, "bootstrap_sft"),
            variants_per_task=variants_per_task,
            num_train_epochs=bootstrap_num_train_epochs,
            learning_rate=bootstrap_learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=bootstrap_per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=bootstrap_max_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            seed=seed,
            max_new_tokens=max_completion_length,
            showcase_limit=bootstrap_showcase_limit,
            training_data_mode=bootstrap_training_data_mode,
            teacher_min_score=bootstrap_teacher_min_score,
            teacher_max_penalties=bootstrap_teacher_max_penalties,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.use_cache = False

    peft_config = None
    if resolved_bootstrap_path:
        adapter_config_path = os.path.join(resolved_bootstrap_path, "adapter_config.json")
        if not os.path.isdir(resolved_bootstrap_path) or not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(
                "Bootstrap adapter path is invalid. "
                f"Expected a local PEFT adapter directory containing adapter_config.json, got: {resolved_bootstrap_path}"
            )
        model = PeftModel.from_pretrained(model, resolved_bootstrap_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
        )

    training_args = _build_grpo_config(
        GRPOConfig=GRPOConfig,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        generation_batch_size=max(per_device_train_batch_size, num_generations),
        temperature=temperature,
        top_p=top_p,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "reward_funcs": EnvironmentRewardFunction(
            reward_mode=reward_mode,
            summary_bonus_scale=summary_bonus_scale,
            reply_bonus_scale=reply_bonus_scale,
            exact_action_bonus_scale=exact_action_bonus_scale,
            acceptable_action_penalty_scale=acceptable_action_penalty_scale,
        ),
        "train_dataset": Dataset.from_list(train_rows),
    }
    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if peft_config is not None and "peft_config" in trainer_signature.parameters:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    trained_policy = HuggingFaceModelPolicy(
        base_model=base_model,
        adapter_path=output_dir,
        max_new_tokens=max_completion_length,
        temperature=0.0,
    )
    trained_eval = evaluate_policy(trained_policy, eval_tasks)
    save_evaluation(os.path.join(output_dir, "trained_eval.json"), trained_eval)

    reward_curve = _build_reward_curve(base_model, output_dir, eval_tasks, max_completion_length)
    loss_curve = _extract_loss_curve(getattr(trainer.state, "log_history", []))
    write_line_chart_svg(os.path.join(output_dir, "reward_curve.svg"), reward_curve, "Evaluation Reward By Epoch", "Avg Reward")
    write_line_chart_svg(os.path.join(output_dir, "loss_curve.svg"), loss_curve, "Training Loss", "Loss")

    with open(os.path.join(output_dir, "trainer_log_history.json"), "w", encoding="utf-8") as handle:
        json.dump(getattr(trainer.state, "log_history", []), handle, indent=2)

    summary = {
        "training_type": "grpo_env_reward",
        "output_dir": output_dir,
        "base_model": base_model,
        "adapter_path": output_dir,
        "bootstrap_adapter_path": resolved_bootstrap_path,
        "bootstrap_auto": bootstrap_auto,
        "reward_mode": reward_mode,
        "summary_bonus_scale": summary_bonus_scale,
        "reply_bonus_scale": reply_bonus_scale,
        "exact_action_bonus_scale": exact_action_bonus_scale,
        "acceptable_action_penalty_scale": acceptable_action_penalty_scale,
        "variants_per_task": variants_per_task,
        "train_case_count": len(train_rows),
        "eval_case_count": len(eval_tasks),
        "split_summary": split_summary,
        "baseline_metrics": baseline_eval["metrics"],
        "trained_metrics": trained_eval["metrics"],
        "heuristic_metrics": heuristic_eval["metrics"],
        "improvement": _metric_delta(baseline_eval["metrics"], trained_eval["metrics"]),
        "gap_to_teacher": _metric_delta(trained_eval["metrics"], heuristic_eval["metrics"]),
    }
    if bootstrap_summary is not None:
        summary["bootstrap_summary"] = {
            "training_type": bootstrap_summary.get("training_type"),
            "output_dir": bootstrap_summary.get("output_dir"),
            "adapter_path": bootstrap_summary.get("adapter_path"),
            "train_case_count": bootstrap_summary.get("train_case_count"),
            "eval_case_count": bootstrap_summary.get("eval_case_count"),
            "schema_improvement": bootstrap_summary.get("schema_improvement"),
            "trained_schema_metrics": bootstrap_summary.get("trained_schema_metrics"),
        }
    save_json(os.path.join(output_dir, "training_summary.json"), summary)
    return summary


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


def _build_grpo_config(GRPOConfig, **kwargs):
    supported = set(inspect.signature(GRPOConfig).parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return GRPOConfig(**filtered)


def _prepare_bootstrap_adapter(
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
) -> tuple[str, dict | None]:
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if os.path.isdir(output_dir) and os.path.isfile(adapter_config_path):
        return output_dir, None

    summary = run_phase55_bootstrap_sft(
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
    return str(summary["adapter_path"]), summary


def _training_imports():
    try:
        from pathlib import Path

        original_read_text = Path.read_text

        def read_text_utf8(self, encoding=None, errors=None):
            return original_read_text(self, encoding=encoding or "utf-8", errors=errors)

        Path.read_text = read_text_utf8

        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Phase 6 RL dependencies are missing. Install torch, transformers, datasets, peft, accelerate, and trl."
        ) from exc

    return torch, Dataset, AutoModelForCausalLM, AutoTokenizer, GRPOConfig, GRPOTrainer, LoraConfig, TaskType, PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 environment-connected GRPO training for PhysioSupportEnv.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to optimize.")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase6/grpo_training", help="Directory for checkpoints and reports.")
    parser.add_argument("--variants-per-task", type=int, default=8, help="Synthetic variants created per base task.")
    parser.add_argument("--num-train-epochs", type=float, default=2.0, help="GRPO training epochs.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Maximum prompt token length.")
    parser.add_argument("--max-completion-length", type=int, default=256, help="Maximum generated completion length.")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions sampled per prompt for GRPO.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature during GRPO rollouts.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling for GRPO rollouts.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when training from the base model directly.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha when training from the base model directly.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout when training from the base model directly.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["raw", "task_score", "shaped"],
        default="shaped",
        help="Reward signal to optimize from the environment.",
    )
    parser.add_argument(
        "--summary-bonus-scale",
        type=float,
        default=0.10,
        help="Extra shaped reward added when therapist_summary matches task coverage well.",
    )
    parser.add_argument(
        "--reply-bonus-scale",
        type=float,
        default=0.05,
        help="Extra shaped reward added when patient_reply is aligned, safe, and concise.",
    )
    parser.add_argument(
        "--exact-action-bonus-scale",
        type=float,
        default=0.02,
        help="Small shaped reward added when the chosen next_action matches the canonical truth action exactly.",
    )
    parser.add_argument(
        "--acceptable-action-penalty-scale",
        type=float,
        default=0.01,
        help="Very small shaped penalty for choosing an acceptable fallback action instead of the canonical truth action.",
    )
    parser.add_argument(
        "--bootstrap-adapter-path",
        type=str,
        default="",
        help="Optional warm-start adapter path from Phase 5 or teacher-SFT before RL optimization.",
    )
    parser.add_argument(
        "--bootstrap-auto",
        action="store_true",
        help="Run the Phase 5.5 bootstrap SFT stage first when no bootstrap adapter path is provided.",
    )
    parser.add_argument(
        "--bootstrap-output-dir",
        type=str,
        default="",
        help="Where to save the bootstrap SFT adapter. Defaults to <output-dir>/bootstrap_sft.",
    )
    parser.add_argument("--bootstrap-num-train-epochs", type=float, default=2.0, help="Bootstrap SFT training epochs.")
    parser.add_argument("--bootstrap-learning-rate", type=float, default=2e-4, help="Bootstrap SFT learning rate.")
    parser.add_argument(
        "--bootstrap-per-device-eval-batch-size",
        type=int,
        default=1,
        help="Bootstrap SFT per-device eval batch size.",
    )
    parser.add_argument("--bootstrap-max-length", type=int, default=1024, help="Bootstrap SFT maximum tokenized sequence length.")
    parser.add_argument(
        "--bootstrap-showcase-limit",
        type=int,
        default=3,
        help="How many representative bootstrap examples to save.",
    )
    parser.add_argument(
        "--bootstrap-training-data-mode",
        type=str,
        choices=["structured", "teacher", "hybrid"],
        default="hybrid",
        help="Bootstrap SFT source used by --bootstrap-auto.",
    )
    parser.add_argument(
        "--bootstrap-teacher-min-score",
        type=float,
        default=0.8,
        help="Minimum teacher task score required when bootstrap-training-data-mode=teacher.",
    )
    parser.add_argument(
        "--bootstrap-teacher-max-penalties",
        type=int,
        default=0,
        help="Maximum number of teacher penalties allowed when bootstrap-training-data-mode=teacher.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_phase6_training(
        base_model=args.base_model,
        output_dir=args.output_dir,
        variants_per_task=args.variants_per_task,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        reward_mode=args.reward_mode,
        summary_bonus_scale=args.summary_bonus_scale,
        reply_bonus_scale=args.reply_bonus_scale,
        exact_action_bonus_scale=args.exact_action_bonus_scale,
        acceptable_action_penalty_scale=args.acceptable_action_penalty_scale,
        bootstrap_adapter_path=args.bootstrap_adapter_path,
        bootstrap_auto=args.bootstrap_auto,
        bootstrap_output_dir=args.bootstrap_output_dir,
        bootstrap_num_train_epochs=args.bootstrap_num_train_epochs,
        bootstrap_learning_rate=args.bootstrap_learning_rate,
        bootstrap_per_device_eval_batch_size=args.bootstrap_per_device_eval_batch_size,
        bootstrap_max_length=args.bootstrap_max_length,
        bootstrap_showcase_limit=args.bootstrap_showcase_limit,
        bootstrap_training_data_mode=args.bootstrap_training_data_mode,
        bootstrap_teacher_min_score=args.bootstrap_teacher_min_score,
        bootstrap_teacher_max_penalties=args.bootstrap_teacher_max_penalties,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
