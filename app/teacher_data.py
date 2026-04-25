from __future__ import annotations

import json
from copy import deepcopy

from app.env import PhysioSupportEnv
from app.grader import grade_episode
from app.heuristic_policy import HeuristicPolicy
from app.prompting import build_messages, render_plain_training_text
from app.training_data import build_sft_splits


def build_teacher_training_bundle(
    variants_per_task: int = 8,
    min_score: float = 0.8,
    max_penalties: int = 0,
) -> dict:
    train_examples, eval_examples = build_sft_splits(variants_per_task=variants_per_task)
    teacher = HeuristicPolicy()

    train_teacher_examples: list[dict] = []
    train_teacher_records: list[dict] = []
    eval_teacher_records: list[dict] = []

    for example in train_examples:
        record = _score_teacher_example(example, teacher)
        if record["score"] >= min_score and not record["unsafe"] and len(record["penalties"]) <= max_penalties:
            record["accepted"] = True
            train_teacher_examples.append(_teacher_example_from_record(example, record))
        train_teacher_records.append(record)

    for example in eval_examples:
        eval_teacher_records.append(_score_teacher_example(example, teacher))

    summary = {
        "variants_per_task": variants_per_task,
        "min_score": min_score,
        "max_penalties": max_penalties,
        "train_source_count": len(train_examples),
        "train_teacher_count": len(train_teacher_examples),
        "eval_source_count": len(eval_examples),
        "train_accept_rate": _ratio(len(train_teacher_examples), len(train_examples)),
        "train_avg_teacher_score": _average([record["score"] for record in train_teacher_records]),
        "train_avg_teacher_reward": _average([record["reward"] for record in train_teacher_records]),
        "eval_avg_teacher_score": _average([record["score"] for record in eval_teacher_records]),
        "eval_avg_teacher_reward": _average([record["reward"] for record in eval_teacher_records]),
        "train_unsafe_count": sum(1 for record in train_teacher_records if record["unsafe"]),
        "eval_unsafe_count": sum(1 for record in eval_teacher_records if record["unsafe"]),
    }

    return {
        "train_examples": train_teacher_examples,
        "eval_tasks": [dict(example["task"]) for example in eval_examples],
        "train_records": train_teacher_records,
        "eval_records": eval_teacher_records,
        "summary": summary,
    }


def write_teacher_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _score_teacher_example(example: dict, teacher: HeuristicPolicy) -> dict:
    task = dict(example["task"])
    observation = dict(example["observation"])
    decision = teacher.predict(observation)

    env = PhysioSupportEnv(task=task)
    env.reset_dict()
    _, reward, _, info = env.step_dict(decision)
    raw_reward = float(info.get("raw_total_reward", reward))
    score = float(info.get("task_score", grade_episode(raw_reward, env.state_dict())))

    return {
        "task_id": task["task_id"],
        "base_task_id": task.get("base_task_id", task["task_id"]),
        "task_family": task["task_family"],
        "decision": decision,
        "reward": raw_reward,
        "score": score,
        "penalties": list(info.get("penalties", [])),
        "unsafe": bool(info.get("unsafe", False)),
        "accepted": False,
        "breakdown": dict(info.get("breakdown", {})),
    }


def _teacher_example_from_record(example: dict, record: dict) -> dict:
    observation = deepcopy(example["observation"])
    decision = deepcopy(record["decision"])
    messages = build_messages(observation, decision)
    return {
        "task_id": example["task_id"],
        "base_task_id": example["base_task_id"],
        "split": example["split"],
        "variant_id": example["variant_id"],
        "variant_profile": deepcopy(example.get("variant_profile", {})),
        "task_family": example["task_family"],
        "task": deepcopy(example["task"]),
        "observation": observation,
        "messages": messages,
        "text": render_plain_training_text(observation, decision),
        "teacher_decision": decision,
        "teacher_score": record["score"],
        "teacher_reward": record["reward"],
        "teacher_penalties": list(record["penalties"]),
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
