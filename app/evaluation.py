from __future__ import annotations

import json
from statistics import mean

from app.env import PhysioSupportEnv
from app.grader import grade_episode

_CALLBACK_ACTIONS = {
    "schedule_callback",
    "priority_callback",
    "notify_therapist",
    "escalate_for_clinical_review",
    "escalate_for_emergency_attention",
}
_PRIORITY_ACTIONS = {
    "priority_callback",
    "escalate_for_clinical_review",
    "escalate_for_emergency_attention",
}


def evaluate_policy(policy, tasks: list[dict]) -> dict:
    results = [run_case(policy, task) for task in tasks]
    metrics = compute_metrics(results)
    return {"metrics": metrics, "results": results}


def run_case(policy, task: dict) -> dict:
    env = PhysioSupportEnv(task=task)
    observation = env.reset_dict()
    decision_error = None

    try:
        decision = policy.predict(observation)
    except Exception as exc:
        decision = {"policy_error": f"{exc.__class__.__name__}: {exc}"}
        decision_error = str(exc)

    state, reward, done, info = env.step_dict(decision)
    raw_reward = float(info.get("raw_total_reward", reward))
    score = float(info.get("task_score", grade_episode(raw_reward, env.state_dict())))
    truth = task["truth"]

    return {
        "task_id": task["task_id"],
        "base_task_id": task.get("base_task_id", task["task_id"]),
        "task_family": task["task_family"],
        "reward": raw_reward,
        "score": score,
        "done": done,
        "unsafe": bool(info.get("unsafe", False)),
        "decision": decision,
        "decision_error": decision_error,
        "state": state,
        "breakdown": info.get("breakdown", {}),
        "penalties": info.get("penalties", []),
        "truth": {
            "intent": truth["intent"],
            "risk_level": truth["risk_level"],
            "next_action": truth["next_action"],
        },
        "intent_exact": decision.get("intent") == truth["intent"],
        "risk_exact": decision.get("risk_level") == truth["risk_level"],
        "action_exact": decision.get("next_action") == truth["next_action"],
        "callback_correct": _callback_correct(task, decision),
        "priority_correct": _priority_correct(task, decision),
        "summary_ratio": _safe_ratio(info.get("breakdown", {}).get("summary_completeness", 0.0), 0.05),
    }


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {
            "num_cases": 0,
            "avg_reward": 0.0,
            "avg_score": 0.0,
            "intent_accuracy": 0.0,
            "risk_accuracy": 0.0,
            "action_accuracy": 0.0,
            "callback_correctness": 0.0,
            "priority_pain_recall": 0.0,
            "unsafe_action_rate": 0.0,
            "summary_completeness": 0.0,
            "component_averages": {},
        }

    component_names = sorted({name for result in results for name in result["breakdown"].keys()})
    callback_cases = [result for result in results if _is_callback_case(result)]
    priority_cases = [result for result in results if _is_priority_case(result)]

    return {
        "num_cases": len(results),
        "avg_reward": mean(result["reward"] for result in results),
        "avg_score": mean(result["score"] for result in results),
        "intent_accuracy": mean(1.0 if result["intent_exact"] else 0.0 for result in results),
        "risk_accuracy": mean(1.0 if result["risk_exact"] else 0.0 for result in results),
        "action_accuracy": mean(1.0 if result["action_exact"] else 0.0 for result in results),
        "callback_correctness": mean(1.0 if result["callback_correct"] else 0.0 for result in callback_cases)
        if callback_cases
        else 0.0,
        "priority_pain_recall": mean(1.0 if result["priority_correct"] else 0.0 for result in priority_cases)
        if priority_cases
        else 0.0,
        "unsafe_action_rate": mean(1.0 if result["unsafe"] else 0.0 for result in results),
        "summary_completeness": mean(result["summary_ratio"] for result in results),
        "component_averages": {
            name: mean(result["breakdown"].get(name, 0.0) for result in results) for name in component_names
        },
    }


def save_evaluation(path: str, evaluation: dict) -> None:
    save_json(path, evaluation)


def save_json(path: str, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_task_manifest(tasks: list[dict]) -> list[dict]:
    manifest: list[dict] = []
    for task in tasks:
        truth = task["truth"]
        manifest.append(
            {
                "task_id": task["task_id"],
                "base_task_id": task.get("base_task_id", task["task_id"]),
                "split": task.get("split", "unknown"),
                "variant_id": task.get("variant_id"),
                "task_family": task["task_family"],
                "patient_message": task["patient_message"],
                "allowed_actions": list(task["allowed_actions"]),
                "visit_context": dict(task["visit_context"]),
                "appointment_context": dict(task["appointment_context"]),
                "operational_constraints": list(task["operational_constraints"]),
                "policy_constraints": list(task["policy_constraints"]),
                "success_score_threshold": float(task.get("success_score_threshold", 0.72)),
                "truth": {
                    "intent": truth["intent"],
                    "risk_level": truth["risk_level"],
                    "next_action": truth["next_action"],
                    "risk_flag": truth["risk_flag"],
                },
            }
        )
    return manifest


def build_showcase_examples(results: list[dict], limit: int = 3) -> list[dict]:
    selected: list[dict] = []
    seen_task_ids: set[str] = set()
    seen_base_task_ids: set[str] = set()
    candidate_groups = [
        [result for result in results if not result["action_exact"]],
        [result for result in results if not result["intent_exact"]],
        [result for result in results if result["task_family"] == "priority_pain"],
        [result for result in results if result["task_family"] == "callback"],
        list(results),
    ]

    for group in candidate_groups:
        for result in sorted(group, key=_result_sort_key):
            if result["task_id"] in seen_task_ids or result["base_task_id"] in seen_base_task_ids:
                continue
            selected.append(_format_showcase_example(result))
            seen_task_ids.add(result["task_id"])
            seen_base_task_ids.add(result["base_task_id"])
            if len(selected) >= limit:
                return selected

    for result in sorted(results, key=_result_sort_key):
        if result["task_id"] in seen_task_ids:
            continue
        selected.append(_format_showcase_example(result))
        seen_task_ids.add(result["task_id"])
        if len(selected) >= limit:
            return selected

    return selected


def _callback_correct(task: dict, decision: dict) -> bool:
    truth = task["truth"]
    needs_callback = truth["intent"] in {"request_callback", "report_worsening_pain", "mixed_intent"} or truth.get(
        "escalation_required", False
    )
    if not needs_callback:
        return False
    return decision.get("next_action") in _CALLBACK_ACTIONS


def _priority_correct(task: dict, decision: dict) -> bool:
    truth = task["truth"]
    if truth["risk_level"] not in {"high", "critical"}:
        return False
    return decision.get("next_action") in _PRIORITY_ACTIONS


def _is_callback_case(result: dict) -> bool:
    return result["task_family"] in {"callback", "priority_pain"}


def _is_priority_case(result: dict) -> bool:
    return result["truth"]["risk_level"] in {"high", "critical"}


def _safe_ratio(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))


def _format_showcase_example(result: dict) -> dict:
    state = result["state"]
    return {
        "selection_reason": _selection_reason(result),
        "task_id": result["task_id"],
        "base_task_id": result["base_task_id"],
        "task_family": result["task_family"],
        "patient_message": state["patient_message"],
        "allowed_actions": list(state["allowed_actions"]),
        "score": result["score"],
        "reward": result["reward"],
        "unsafe": result["unsafe"],
        "decision": result["decision"],
        "truth": result["truth"],
        "intent_exact": result["intent_exact"],
        "risk_exact": result["risk_exact"],
        "action_exact": result["action_exact"],
        "callback_correct": result["callback_correct"],
        "priority_correct": result["priority_correct"],
        "penalties": list(result["penalties"]),
        "breakdown": dict(result["breakdown"]),
    }


def _selection_reason(result: dict) -> str:
    if not result["action_exact"]:
        return "action_miss"
    if not result["intent_exact"]:
        return "intent_miss"
    if result["task_family"] == "priority_pain":
        return "priority_case"
    if result["task_family"] == "callback":
        return "callback_case"
    return "representative_case"


def _result_sort_key(result: dict) -> tuple[str, str]:
    return (result["base_task_id"], result["task_id"])
