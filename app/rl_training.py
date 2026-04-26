from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.env import PhysioSupportEnv
from app.grader import grade_episode
from app.prompting import build_messages
from app.structured_output import extract_json_object, validate_submission_payload
from app.training_data import build_sft_splits, build_split_summary


def build_grpo_training_bundle(variants_per_task: int = 8) -> dict:
    train_examples, eval_examples = build_sft_splits(variants_per_task=variants_per_task)
    train_rows = [_build_grpo_row(example) for example in train_examples]
    eval_tasks = [deepcopy(example["task"]) for example in eval_examples]
    split_summary = build_split_summary(variants_per_task=variants_per_task)
    return {
        "train_rows": train_rows,
        "eval_tasks": eval_tasks,
        "split_summary": split_summary,
    }


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def score_completion_text(task: dict, observation: dict, completion_text: str) -> dict:
    env = PhysioSupportEnv(task=deepcopy(task))
    env.reset_dict()

    try:
        action_input: dict[str, Any] = extract_json_object(completion_text)
        action_input = validate_submission_payload(action_input, observation)
    except Exception:
        action_input = {"raw_completion": completion_text}

    _, reward, _, info = env.step_dict(action_input)
    raw_reward = float(info.get("raw_total_reward", reward))
    task_score = float(info.get("task_score", grade_episode(raw_reward, env.state_dict())))

    return {
        "raw_reward": raw_reward,
        "task_score": task_score,
        "unsafe": bool(info.get("unsafe", False)),
        "penalties": list(info.get("penalties", [])),
        "breakdown": dict(info.get("breakdown", {})),
    }


@dataclass
class EnvironmentRewardFunction:
    reward_mode: str = "raw"

    def __post_init__(self) -> None:
        self.__name__ = f"environment_reward_{self.reward_mode}"

    def __call__(self, completions, task_json, observation_json, **kwargs) -> list[float]:
        rewards: list[float] = []
        for completion, serialized_task, serialized_observation in zip(completions, task_json, observation_json):
            task = json.loads(serialized_task) if isinstance(serialized_task, str) else deepcopy(serialized_task)
            observation = (
                json.loads(serialized_observation)
                if isinstance(serialized_observation, str)
                else deepcopy(serialized_observation)
            )
            completion_text = _completion_to_text(completion)
            result = score_completion_text(task, observation, completion_text)
            reward_key = "task_score" if self.reward_mode == "task_score" else "raw_reward"
            rewards.append(float(result[reward_key]))
        return rewards


def _build_grpo_row(example: dict) -> dict:
    observation = deepcopy(example["observation"])
    task = deepcopy(example["task"])
    return {
        "prompt": build_messages(observation),
        "task_json": json.dumps(task, ensure_ascii=True),
        "observation_json": json.dumps(observation, ensure_ascii=True),
        "task_id": example["task_id"],
        "base_task_id": example["base_task_id"],
        "task_family": example["task_family"],
        "split": example["split"],
        "variant_id": example["variant_id"],
        "variant_profile_json": json.dumps(example.get("variant_profile", {}), ensure_ascii=True),
    }


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content

    if isinstance(completion, list):
        text_parts: list[str] = []
        for item in completion:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                text_parts.append(item["content"])
        if text_parts:
            return "\n".join(text_parts)

    return str(completion)
