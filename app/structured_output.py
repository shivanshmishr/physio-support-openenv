from __future__ import annotations

import json
import re
from typing import get_args

from app.models import CareCoordinationOutput, NextActionName

_ALL_ACTIONS = set(get_args(NextActionName))
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_INTENT_ALIASES = {
    "confirmation_of_home_visit": "book_visit",
    "confirm_home_visit": "book_visit",
    "book_home_visit": "book_visit",
    "reschedule_home_visit": "reschedule_visit",
    "priority_callback": "request_callback",
    "schedule_callback": "request_callback",
    "emergency_attention": "report_worsening_pain",
    "urgent_pain_case": "report_worsening_pain",
}
_RISK_LEVEL_ALIASES = {
    "routine": "low",
    "stable": "low",
    "moderate": "medium",
    "urgent": "high",
    "emergency": "critical",
}
_NEXT_ACTION_ALIASES = {
    "confirm_home_visit": "confirm_home_visit",
    "reschedule_home_visit": "reschedule_home_visit",
    "schedule_callback": "schedule_callback",
    "priority_callback": "priority_callback",
    "notify_therapist": "notify_therapist",
    "modify_visit_plan": "modify_visit_plan",
    "convert_to_remote_checkin": "convert_to_remote_checkin",
    "request_more_information": "request_more_information",
    "escalate_for_clinical_review": "escalate_for_clinical_review",
    "escalate_for_emergency_attention": "escalate_for_emergency_attention",
    "close_with_guidance": "close_with_guidance",
}


def extract_json_object(content: str) -> dict:
    cleaned = _THINK_PATTERN.sub("", content).strip()

    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    extracted = _extract_first_json_object(cleaned)
    if extracted is None:
        raise ValueError(f"Model response did not contain a valid JSON object: {content}")

    return extracted


def _extract_first_json_object(content: str) -> dict | None:
    for start_index, char in enumerate(content):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escape = False

        for end_index in range(start_index, len(content)):
            current = content[end_index]

            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start_index : end_index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return None


def validate_submission_payload(payload: dict, observation: dict) -> dict:
    normalized = normalize_submission_payload(payload, observation)
    submission = CareCoordinationOutput.model_validate(normalized)
    allowed_actions = set(observation.get("allowed_actions", []))

    if submission.next_action not in allowed_actions:
        raise ValueError(
            f"Invalid next_action returned by model: {submission.next_action}. Allowed actions: {sorted(allowed_actions)}"
        )

    invalid_secondary = [action for action in submission.secondary_actions if action not in _ALL_ACTIONS]
    if invalid_secondary:
        raise ValueError(f"Invalid secondary_actions returned by model: {invalid_secondary}")

    normalized_secondary = [
        action for action in dict.fromkeys(submission.secondary_actions) if action != submission.next_action
    ]

    return submission.model_copy(update={"secondary_actions": normalized_secondary}).model_dump()


def normalize_submission_payload(payload: dict, observation: dict) -> dict:
    normalized = dict(payload)
    message = observation.get("patient_message", "").lower()
    allowed_actions = set(observation.get("allowed_actions", []))
    task_family = observation.get("task_family", "")

    normalized["secondary_actions"] = _normalize_secondary_actions(normalized.get("secondary_actions"))
    normalized["next_action"] = _normalize_next_action(normalized.get("next_action"), allowed_actions, message, task_family)
    normalized["risk_level"] = _normalize_risk_level(normalized.get("risk_level"), normalized["next_action"], message)
    normalized["intent"] = _normalize_intent(normalized.get("intent"), normalized["next_action"], message)
    normalized.setdefault("patient_reply", "")
    normalized.setdefault("therapist_summary", "")
    normalized.setdefault("risk_flag", None)
    return normalized


def _normalize_secondary_actions(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip() in _ALL_ACTIONS]
    return []


def _normalize_next_action(value, allowed_actions: set[str], message: str, task_family: str) -> str:
    if isinstance(value, str):
        candidate = _NEXT_ACTION_ALIASES.get(value.strip(), value.strip())
        if candidate in allowed_actions:
            return candidate

    has_critical = any(token in message for token in ["emergency", "extreme pain", "chest tightness", "urgent help right now"])
    has_pain = any(token in message for token in ["pain", "worsening", "8 out of 10", "severe"])
    has_callback = any(token in message for token in ["call", "callback"])
    has_reschedule = any(token in message for token in ["reschedule", "move it", "move tomorrow's", "tomorrow afternoon"])

    if "escalate_for_emergency_attention" in allowed_actions and has_critical:
        return "escalate_for_emergency_attention"

    if task_family == "callback" and "schedule_callback" in allowed_actions:
        if has_callback:
            return "schedule_callback"
        if has_pain and "priority_callback" in allowed_actions:
            return "priority_callback"

    if task_family == "rescheduling" and "reschedule_home_visit" in allowed_actions and has_reschedule:
        return "reschedule_home_visit"

    if task_family == "priority_pain" and "priority_callback" in allowed_actions and (has_pain or has_callback):
        return "priority_callback"

    if "priority_callback" in allowed_actions and has_pain:
        return "priority_callback"
    if "schedule_callback" in allowed_actions and has_callback:
        return "schedule_callback"
    if "reschedule_home_visit" in allowed_actions and any(
        token in message for token in ["reschedule", "move it", "caregiver", "lift is out", "stairs"]
    ):
        return "reschedule_home_visit"
    if "confirm_home_visit" in allowed_actions:
        return "confirm_home_visit"
    if allowed_actions:
        return sorted(allowed_actions)[0]
    return "request_more_information"


def _normalize_risk_level(value, next_action: str, message: str) -> str:
    if isinstance(value, str):
        candidate = _RISK_LEVEL_ALIASES.get(value.strip().lower(), value.strip().lower())
        if candidate in {"low", "medium", "high", "critical"}:
            value = candidate
        else:
            value = None
    else:
        value = None

    if next_action == "escalate_for_emergency_attention":
        return "critical"
    if next_action in {"priority_callback", "escalate_for_clinical_review"}:
        return "high"
    if next_action in {"reschedule_home_visit", "modify_visit_plan", "convert_to_remote_checkin"}:
        return "medium"
    if next_action == "confirm_home_visit":
        return "low"
    if value is not None:
        return value
    if any(token in message for token in ["extreme pain", "chest tightness", "urgent help right now"]):
        return "critical"
    if any(token in message for token in ["pain", "worsening", "8 out of 10", "severe"]):
        return "high"
    if any(token in message for token in ["caregiver", "stairs", "lift is out", "reschedule", "move it"]):
        return "medium"
    return "low"


def _normalize_intent(value, next_action: str, message: str) -> str:
    if isinstance(value, str):
        candidate = _INTENT_ALIASES.get(value.strip(), value.strip())
        if candidate in set(get_args(CareCoordinationOutput.model_fields["intent"].annotation)):
            value = candidate
        else:
            value = None
    else:
        value = None

    if value is not None:
        return value

    has_pain = any(token in message for token in ["pain", "worsening", "tightness", "8 out of 10", "severe"])
    has_callback = any(token in message for token in ["call", "callback"])
    has_reschedule = any(token in message for token in ["reschedule", "move it", "tomorrow afternoon", "visit moved"])
    has_caregiver = "caregiver" in message
    has_access = any(token in message for token in ["stairs", "lift is out", "access issue"])

    if next_action == "confirm_home_visit":
        return "book_visit"
    if next_action == "reschedule_home_visit":
        if has_caregiver:
            return "caregiver_unavailable"
        if has_access:
            return "home_access_issue"
        return "reschedule_visit"
    if next_action in {"schedule_callback", "notify_therapist"}:
        if has_pain and (has_reschedule or has_access or has_caregiver):
            return "mixed_intent"
        if has_pain:
            return "report_worsening_pain"
        return "request_callback"
    if next_action in {"priority_callback", "escalate_for_clinical_review", "escalate_for_emergency_attention"}:
        if has_reschedule or has_access or has_caregiver:
            return "mixed_intent"
        return "report_worsening_pain"
    if has_caregiver:
        return "caregiver_unavailable"
    if has_access:
        return "home_access_issue"
    if has_reschedule:
        return "reschedule_visit"
    return "book_visit"
