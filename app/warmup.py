from __future__ import annotations

from statistics import mean

from app.env import PhysioSupportEnv
from app.grader import grade_episode
from app.training_data import task_to_observation

_REQUIRED_KEYS = (
    "intent",
    "risk_level",
    "next_action",
    "secondary_actions",
    "patient_reply",
    "therapist_summary",
    "risk_flag",
)


def evaluate_warmup_policy(policy, tasks: list[dict]) -> dict:
    results = [run_warmup_case(policy, task) for task in tasks]
    return {
        "metrics": compute_warmup_metrics(results),
        "results": results,
    }


def run_warmup_case(policy, task: dict) -> dict:
    observation = task_to_observation(task)
    prediction_error = None
    decision = None

    try:
        decision = policy.predict(observation)
    except Exception as exc:
        prediction_error = f"{exc.__class__.__name__}: {exc}"

    schema_valid = decision is not None
    required_keys_present = schema_valid and all(key in decision for key in _REQUIRED_KEYS)
    allowed_action_valid = schema_valid and decision["next_action"] in set(observation.get("allowed_actions", []))
    secondary_actions_valid = schema_valid and isinstance(decision.get("secondary_actions"), list)

    score = None
    reward = None
    penalties: list[str] = []

    if schema_valid:
        env = PhysioSupportEnv(task=task)
        env.reset_dict()
        _, env_reward, _, info = env.step_dict(decision)
        reward = float(info.get("raw_total_reward", env_reward))
        score = float(info.get("task_score", grade_episode(reward, env.state_dict())))
        penalties = list(info.get("penalties", []))

    return {
        "task_id": task["task_id"],
        "base_task_id": task.get("base_task_id", task["task_id"]),
        "task_family": task["task_family"],
        "schema_valid": schema_valid,
        "required_keys_present": required_keys_present,
        "allowed_action_valid": allowed_action_valid,
        "secondary_actions_valid": secondary_actions_valid,
        "prediction_error": prediction_error,
        "decision": decision,
        "score": score,
        "reward": reward,
        "penalties": penalties,
        "allowed_actions": list(observation.get("allowed_actions", [])),
        "patient_message": observation["patient_message"],
        "truth": {
            "intent": task["truth"]["intent"],
            "risk_level": task["truth"]["risk_level"],
            "next_action": task["truth"]["next_action"],
        },
    }


def compute_warmup_metrics(results: list[dict]) -> dict:
    if not results:
        return {
            "num_cases": 0,
            "schema_valid_rate": 0.0,
            "required_keys_rate": 0.0,
            "allowed_action_rate": 0.0,
            "secondary_actions_rate": 0.0,
            "avg_score_on_valid": 0.0,
            "avg_reward_on_valid": 0.0,
            "invalid_case_count": 0,
        }

    valid_results = [result for result in results if result["schema_valid"]]

    return {
        "num_cases": len(results),
        "schema_valid_rate": mean(1.0 if result["schema_valid"] else 0.0 for result in results),
        "required_keys_rate": mean(1.0 if result["required_keys_present"] else 0.0 for result in results),
        "allowed_action_rate": mean(1.0 if result["allowed_action_valid"] else 0.0 for result in results),
        "secondary_actions_rate": mean(1.0 if result["secondary_actions_valid"] else 0.0 for result in results),
        "avg_score_on_valid": mean(float(result["score"]) for result in valid_results) if valid_results else 0.0,
        "avg_reward_on_valid": mean(float(result["reward"]) for result in valid_results) if valid_results else 0.0,
        "invalid_case_count": sum(1 for result in results if not result["schema_valid"]),
    }


def build_warmup_showcase(results: list[dict], limit: int = 3) -> list[dict]:
    selected: list[dict] = []
    seen_base_task_ids: set[str] = set()

    candidate_groups = [
        [result for result in results if not result["schema_valid"]],
        [result for result in results if result["schema_valid"] and not result["allowed_action_valid"]],
        [result for result in results if result["task_family"] == "priority_pain"],
        list(results),
    ]

    for group in candidate_groups:
        for result in sorted(group, key=lambda item: (item["base_task_id"], item["task_id"])):
            if result["base_task_id"] in seen_base_task_ids:
                continue
            selected.append(
                {
                    "task_id": result["task_id"],
                    "base_task_id": result["base_task_id"],
                    "task_family": result["task_family"],
                    "schema_valid": result["schema_valid"],
                    "allowed_action_valid": result["allowed_action_valid"],
                    "score": result["score"],
                    "reward": result["reward"],
                    "prediction_error": result["prediction_error"],
                    "decision": result["decision"],
                    "truth": result["truth"],
                    "patient_message": result["patient_message"],
                }
            )
            seen_base_task_ids.add(result["base_task_id"])
            if len(selected) >= limit:
                return selected

    return selected
