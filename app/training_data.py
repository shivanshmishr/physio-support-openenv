from __future__ import annotations

import json
from copy import deepcopy

from app.case_generator import build_case_splits, case_split_summary, validate_case_splits
from app.prompting import build_messages, render_plain_training_text
from app.tasks import TASKS


def build_sft_splits(variants_per_task: int = 8) -> tuple[list[dict], list[dict]]:
    train_cases, eval_cases = build_case_splits(TASKS, variants_per_task=variants_per_task)
    validate_case_splits(train_cases, eval_cases, expected_base_task_ids={task["task_id"] for task in TASKS})
    return build_sft_examples(train_cases), build_sft_examples(eval_cases)


def build_sft_examples(tasks: list[dict]) -> list[dict]:
    examples: list[dict] = []
    for task in tasks:
        observation = task_to_observation(task)
        target = build_target_output(task)
        examples.append(
            {
                "task_id": task["task_id"],
                "base_task_id": task.get("base_task_id", task["task_id"]),
                "split": task.get("split", "unknown"),
                "variant_id": task.get("variant_id"),
                "variant_profile": deepcopy(task.get("variant_profile", {})),
                "task_family": task["task_family"],
                "task": deepcopy(task),
                "observation": observation,
                "messages": build_messages(observation, target),
                "text": render_plain_training_text(observation, target),
                "target_json": json.dumps(target, ensure_ascii=True),
            }
        )
    return examples


def build_split_summary(variants_per_task: int = 8) -> dict:
    train_cases, eval_cases = build_case_splits(TASKS, variants_per_task=variants_per_task)
    validate_case_splits(train_cases, eval_cases, expected_base_task_ids={task["task_id"] for task in TASKS})
    return case_split_summary(train_cases, eval_cases)


def task_to_observation(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "patient_message": task["patient_message"],
        "patient_history_summary": task["patient_history_summary"],
        "care_plan_summary": task["care_plan_summary"],
        "appointment_context": deepcopy(task["appointment_context"]),
        "recent_history": [],
        "visit_context": deepcopy(task["visit_context"]),
        "operational_constraints": list(task["operational_constraints"]),
        "allowed_actions": list(task["allowed_actions"]),
        "policy_constraints": list(task["policy_constraints"]),
        "step_id": 0,
        "max_steps": task["max_steps"],
        "done": False,
        "reward": None,
        "metadata": {},
    }


def build_target_output(task: dict) -> dict:
    truth = task["truth"]
    observation = task_to_observation(task)
    next_action = truth["next_action"]
    secondary_actions = list(truth.get("required_secondary_actions", []))

    if not secondary_actions and truth.get("acceptable_secondary_actions"):
        secondary_actions = truth["acceptable_secondary_actions"][:1]

    patient_reply = _patient_reply(next_action, observation, truth)
    therapist_summary = _therapist_summary(next_action, observation, truth, secondary_actions)

    return {
        "intent": truth["intent"],
        "risk_level": truth["risk_level"],
        "next_action": next_action,
        "secondary_actions": secondary_actions,
        "patient_reply": patient_reply,
        "therapist_summary": therapist_summary,
        "risk_flag": truth["risk_flag"],
    }


def _patient_reply(next_action: str, observation: dict, truth: dict) -> str:
    preferred_window = observation["visit_context"].get("preferred_window", "the requested time")
    scheduled_visit = observation["visit_context"].get("scheduled_visit", "the next visit")

    if next_action == "confirm_home_visit":
        return f"I can help with that. Your home physiotherapy visit is confirmed for {preferred_window}."
    if next_action == "reschedule_home_visit":
        return f"I understand the access and scheduling issue. I will move the home visit to {preferred_window}."
    if next_action == "schedule_callback":
        return f"I'm sorry the pain has worsened. I'm arranging a callback before {scheduled_visit}."
    if next_action == "priority_callback":
        return "I'm sorry the pain has increased. I'm marking this as a priority callback so our team can contact you quickly and review the safest next step."
    if next_action == "escalate_for_clinical_review":
        return "I'm sorry this has worsened. I'm escalating this for urgent clinical review before any further visit planning."
    if next_action == "escalate_for_emergency_attention":
        return "This sounds urgent. Please seek emergency attention immediately while I alert the care team right away."
    if next_action == "modify_visit_plan":
        return "I understand the situation. We will adjust the visit plan and confirm the safest next step shortly."
    if next_action == "convert_to_remote_checkin":
        return "Because of the home access issue, I can switch this to a remote check-in while the team reviews the visit."
    if next_action == "notify_therapist":
        return "I understand the concern. I am notifying the therapist and our team will review this promptly."
    if next_action == "request_more_information":
        return "I want to route this safely. Please share one more detail so I can confirm the next step."
    return "I understand your request. I will share the right guidance and next steps now."


def _therapist_summary(next_action: str, observation: dict, truth: dict, secondary_actions: list[str]) -> str:
    blockers: list[str] = []
    patient_message = observation["patient_message"].lower()
    if "caregiver" in patient_message:
        blockers.append("caregiver unavailable")
    if any(token in patient_message for token in ["stairs", "lift is out", "cannot manage the stairs", "access issue"]):
        blockers.append("home access concern")
    blocker_text = ", ".join(blockers) if blockers else "no major operational blocker"
    secondary_text = ", ".join(secondary_actions) if secondary_actions else "none"
    return (
        f"Patient issue: {truth['intent']}. Risk: {truth['risk_level']}. Next action: {next_action}. "
        f"Operational blockers: {blocker_text}. Secondary actions: {secondary_text}. "
        f"Scheduled context: {observation['visit_context'].get('scheduled_visit', 'new request')}."
    )
