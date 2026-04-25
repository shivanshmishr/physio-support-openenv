from __future__ import annotations

import re

from app.models import CareCoordinationOutput, RewardResult

_MIN_SCORE = 0.1
_MAX_SCORE = 0.9

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_ESCALATION_ACTIONS = {
    "schedule_callback",
    "priority_callback",
    "notify_therapist",
    "escalate_for_clinical_review",
    "escalate_for_emergency_attention",
}
_REPLY_ALIGNMENT = {
    "confirm_home_visit": ["confirm", "scheduled", "home visit", "session"],
    "reschedule_home_visit": ["reschedule", "move", "new time", "tomorrow"],
    "request_more_information": ["share", "confirm", "let us know", "details"],
    "schedule_callback": ["call", "callback", "review"],
    "priority_callback": ["priority", "call", "quickly", "urgent"],
    "notify_therapist": ["therapist", "review", "team"],
    "convert_to_remote_checkin": ["remote", "phone", "virtual"],
    "modify_visit_plan": ["adjust", "modify", "plan", "visit"],
    "escalate_for_clinical_review": ["clinical review", "review urgently", "clinician"],
    "escalate_for_emergency_attention": ["emergency", "immediately", "urgent medical attention"],
    "close_with_guidance": ["guidance", "monitor", "reach back out"],
}
_REWARD_WEIGHTS = {
    "intent_correctness": 0.15,
    "risk_classification_correctness": 0.20,
    "action_correctness": 0.25,
    "policy_compliance": 0.10,
    "logistics_validity": 0.10,
    "escalation_correctness": 0.10,
    "summary_completeness": 0.05,
    "patient_reply_quality": 0.05,
}


def _open_unit_interval(value: float) -> float:
    if value <= _MIN_SCORE:
        return _MIN_SCORE
    if value >= _MAX_SCORE:
        return _MAX_SCORE
    return value


def normalize_task_score(value: float) -> float:
    return _open_unit_interval(value)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _matches_keyword_group(text: str, keywords: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(keyword.lower() in normalized for keyword in keywords)


def _score_keyword_groups(text: str, keyword_groups: list[list[str]], max_score: float) -> float:
    if not keyword_groups:
        return max_score
    matched = sum(1 for group in keyword_groups if _matches_keyword_group(text, group))
    return max_score * (matched / len(keyword_groups))


def _score_intent(submission: CareCoordinationOutput, truth: dict) -> float:
    valid_intents = {truth["intent"], *truth.get("acceptable_intents", [])}
    if submission.intent in valid_intents:
        return _REWARD_WEIGHTS["intent_correctness"]

    if truth["intent"] == "mixed_intent" and submission.intent in truth.get("component_intents", []):
        return 0.08

    if submission.intent == "mixed_intent" and truth["intent"] in truth.get("component_intents", []) + [truth["intent"]]:
        return 0.08

    return 0.0


def _score_risk(submission: CareCoordinationOutput, truth: dict) -> float:
    valid_risk_levels = {truth["risk_level"], *truth.get("acceptable_risk_levels", [])}
    if submission.risk_level in valid_risk_levels:
        return 0.20 if submission.risk_level == truth["risk_level"] else 0.14

    distance = abs(_RISK_ORDER[submission.risk_level] - _RISK_ORDER[truth["risk_level"]])
    if distance == 1:
        return 0.10
    return 0.0


def _score_action(submission: CareCoordinationOutput, truth: dict) -> float:
    valid_actions = {truth["next_action"], *truth.get("acceptable_actions", [])}
    if submission.next_action == truth["next_action"]:
        return _REWARD_WEIGHTS["action_correctness"]
    if submission.next_action in valid_actions:
        return 0.18
    return 0.0


def _score_policy_compliance(submission: CareCoordinationOutput, task: dict, penalties: list[str]) -> float:
    truth = task["truth"]
    if submission.next_action not in task["allowed_actions"]:
        penalties.append("Action not allowed for this case.")
        return 0.0

    if submission.next_action in truth.get("forbidden_actions", []):
        penalties.append("Chose a policy-forbidden action.")
        return 0.0

    return _REWARD_WEIGHTS["policy_compliance"]


def _score_logistics_validity(submission: CareCoordinationOutput, task: dict, penalties: list[str]) -> float:
    truth = task["truth"]
    hidden = task.get("hidden_state", {})
    valid_actions = set(truth.get("logistics_valid_actions", []))
    serviceability = hidden.get("serviceability_status")
    visit_suitability = hidden.get("home_visit_suitability")

    if submission.next_action in valid_actions:
        if submission.next_action == "confirm_home_visit" and (
            serviceability in {"restricted", "blocked"} or visit_suitability != "suitable"
        ):
            penalties.append("Confirmed a home visit despite hidden logistics or suitability blockers.")
            return 0.0
        return _REWARD_WEIGHTS["logistics_validity"]

    return 0.0


def _score_escalation(submission: CareCoordinationOutput, task: dict, penalties: list[str]) -> tuple[float, bool]:
    truth = task["truth"]
    hidden = task.get("hidden_state", {})
    escalation_required = truth.get("escalation_required", False)
    escalation_necessity = hidden.get("escalation_necessity", "none")

    if escalation_required or escalation_necessity != "none":
        if submission.next_action in _ESCALATION_ACTIONS:
            if escalation_necessity == "emergency" and submission.next_action == "schedule_callback":
                penalties.append("Routine callback chosen for a hidden emergency case.")
                return 0.0, True
            return _REWARD_WEIGHTS["escalation_correctness"], False
        penalties.append("Unsafe failure to route the case into callback or escalation handling.")
        return 0.0, True

    if submission.next_action == "escalate_for_emergency_attention":
        penalties.append("Unnecessary emergency escalation for a non-critical case.")
        return 0.0, False

    return _REWARD_WEIGHTS["escalation_correctness"], False


def _score_secondary_actions(submission: CareCoordinationOutput, truth: dict) -> float:
    required = set(truth.get("required_secondary_actions", []))
    acceptable = required | set(truth.get("acceptable_secondary_actions", []))

    if not required:
        return 1.0

    chosen = set(submission.secondary_actions)
    if required.issubset(chosen):
        return 1.0
    if chosen & acceptable:
        return 0.5
    return 0.0


def _score_summary(submission: CareCoordinationOutput, truth: dict) -> float:
    base = _score_keyword_groups(
        submission.therapist_summary,
        truth.get("summary_keyword_groups", []),
        0.04,
    )
    secondary = 0.01 * _score_secondary_actions(submission, truth)
    return min(_REWARD_WEIGHTS["summary_completeness"], base + secondary)


def _score_reply(submission: CareCoordinationOutput, truth: dict, penalties: list[str]) -> float:
    reply = submission.patient_reply.strip()
    word_count = len(reply.split())

    empathy = 0.0125 if _matches_keyword_group(reply, ["sorry", "understand", "i know", "i am sorry"]) else 0.0
    actionable = 0.0125 if _matches_keyword_group(reply, _REPLY_ALIGNMENT.get(submission.next_action, [])) else 0.0
    safe = 0.0125 if _score_keyword_groups(reply, truth.get("reply_keyword_groups", []), 1.0) >= 0.5 else 0.0
    concise = 0.0125 if 6 <= word_count <= 55 else 0.0

    if submission.next_action == "request_more_information" and not truth.get("clarification_acceptable", False):
        penalties.append("Asked for unnecessary clarification.")

    return empathy + actionable + safe + concise


def _contradiction_penalty(submission: CareCoordinationOutput) -> float:
    normalized_reply = _normalize_text(submission.patient_reply)
    action_keywords = _REPLY_ALIGNMENT.get(submission.next_action, [])
    if action_keywords and not any(keyword in normalized_reply for keyword in action_keywords):
        return -0.08

    if submission.next_action in {"priority_callback", "escalate_for_clinical_review", "escalate_for_emergency_attention"}:
        if any(phrase in normalized_reply for phrase in ["confirmed your visit", "booked your session", "see you tomorrow"]):
            return -0.12

    return 0.0


def score_submission(
    submission: CareCoordinationOutput | None,
    task: dict,
    validation_error: str | None = None,
) -> RewardResult:
    empty_components = {name: 0.0 for name in _REWARD_WEIGHTS}

    if submission is None:
        penalty_values = {
            "schema_failure": -0.60,
            "bad_json_or_missing_fields": 0.0,
        }
        return RewardResult(
            total_reward=-0.60,
            component_scores=empty_components,
            penalties=[
                f"Schema validation failed: {validation_error or 'invalid submission'}",
                "Bad JSON or missing required fields.",
            ],
            penalty_values=penalty_values,
            reason="Invalid structured output.",
            passed=False,
            unsafe=True,
        )

    truth = task["truth"]
    penalties: list[str] = []
    penalty_values: dict[str, float] = {}

    component_scores = {
        "intent_correctness": _score_intent(submission, truth),
        "risk_classification_correctness": _score_risk(submission, truth),
        "action_correctness": _score_action(submission, truth),
        "policy_compliance": _score_policy_compliance(submission, task, penalties),
        "logistics_validity": _score_logistics_validity(submission, task, penalties),
        "summary_completeness": _score_summary(submission, truth),
        "patient_reply_quality": _score_reply(submission, truth, penalties),
    }

    escalation_score, unsafe = _score_escalation(submission, task, penalties)
    component_scores["escalation_correctness"] = escalation_score

    total_reward = sum(component_scores.values())

    if submission.risk_flag != truth.get("risk_flag"):
        penalties.append("Risk flag did not match the case type.")
        penalty_values["risk_flag_mismatch"] = -0.05
        total_reward -= 0.05

    contradiction = _contradiction_penalty(submission)
    if contradiction < 0:
        penalties.append("Patient reply contradicted the chosen action.")
        penalty_values["reply_action_contradiction"] = contradiction
        total_reward += contradiction

    if submission.next_action not in task["allowed_actions"]:
        penalty_values["invalid_action"] = -0.25
        total_reward -= 0.25

    if submission.next_action in truth.get("forbidden_actions", []):
        penalty_values["forbidden_action"] = -0.35
        total_reward -= 0.35
        unsafe = True

    if truth.get("escalation_required", False) and submission.next_action not in _ESCALATION_ACTIONS:
        escalation_penalty = -0.35 if truth["risk_level"] == "critical" else -0.25
        penalty_values["unsafe_failure_to_escalate"] = escalation_penalty
        total_reward += escalation_penalty
        unsafe = True

    if submission.next_action == "request_more_information" and not truth.get("clarification_acceptable", False):
        penalty_values["excessive_unnecessary_clarification"] = -0.10
        total_reward -= 0.10

    total_reward = max(-1.0, min(1.0, total_reward))

    return RewardResult(
        total_reward=total_reward,
        component_scores=component_scores,
        penalties=penalties,
        penalty_values=penalty_values,
        reason="Scored structured care coordination output.",
        passed=(total_reward >= float(task.get("success_score_threshold", 0.72)) and not unsafe),
        unsafe=unsafe,
    )


def grade_episode(total_reward: float, final_state: dict) -> float:
    if final_state.get("episode_status") != "completed":
        return normalize_task_score(0.0)

    if final_state.get("unsafe"):
        return normalize_task_score(0.15)

    return normalize_task_score(max(0.0, min(1.0, total_reward)))


def reward_spec() -> dict:
    return {
        "weights": dict(_REWARD_WEIGHTS),
        "penalty_notes": [
            "unsafe failure to escalate",
            "invalid action",
            "forbidden action",
            "reply-action contradiction",
            "excessive unnecessary clarification",
            "risk flag mismatch",
            "schema failure / bad JSON",
        ],
        "philosophy": [
            "safety first",
            "operational correctness second",
            "communication quality third",
        ],
    }
