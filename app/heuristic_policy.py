from __future__ import annotations


def heuristic_decision(observation: dict) -> dict:
    message = observation["patient_message"].lower()
    allowed = observation.get("allowed_actions", [])

    has_reschedule = any(
        token in message
        for token in ["move it", "move the", "reschedule", "change the visit", "change my visit", "move tomorrow's"]
    )
    has_callback = any(token in message for token in ["call", "callback"])
    has_pain = any(token in message for token in ["pain", "tightness", "discomfort", "worse", "worsening"])
    has_access_issue = any(
        token in message for token in ["stairs", "lift is out", "cannot manage the stairs", "access issue"]
    )
    has_caregiver = "caregiver" in message
    is_critical = any(token in message for token in ["chest tightness", "urgent help right now", "extreme pain"])

    intent = "book_visit"
    if has_pain and (has_reschedule or has_access_issue or has_callback):
        intent = "mixed_intent"
    elif has_callback:
        intent = "request_callback"
    elif has_pain:
        intent = "report_worsening_pain"
    elif has_caregiver:
        intent = "caregiver_unavailable"
    elif has_access_issue:
        intent = "home_access_issue"
    elif has_reschedule:
        intent = "reschedule_visit"

    risk_level = "low"
    if is_critical:
        risk_level = "critical"
    elif has_pain and ("8 out of 10" in message or "worse" in message or "worsening" in message):
        risk_level = "high"
    elif has_reschedule or has_access_issue or has_caregiver:
        risk_level = "medium"

    next_action = "request_more_information"
    if is_critical and "escalate_for_emergency_attention" in allowed:
        next_action = "escalate_for_emergency_attention"
    elif risk_level == "high" and "priority_callback" in allowed:
        next_action = "priority_callback"
    elif has_callback and "schedule_callback" in allowed:
        next_action = "schedule_callback"
    elif has_reschedule and "reschedule_home_visit" in allowed:
        next_action = "reschedule_home_visit"
    elif "confirm_home_visit" in allowed:
        next_action = "confirm_home_visit"
    elif allowed:
        next_action = allowed[0]

    secondary_actions: list[str] = []
    if next_action in {"schedule_callback", "priority_callback", "escalate_for_emergency_attention"}:
        if "notify_therapist" in allowed:
            secondary_actions.append("notify_therapist")
    if has_access_issue and "modify_visit_plan" in allowed:
        secondary_actions.append("modify_visit_plan")

    if next_action == "confirm_home_visit":
        patient_reply = "I can help with that. I am confirming your home physiotherapy visit for tomorrow evening."
    elif next_action == "reschedule_home_visit":
        patient_reply = "I understand the caregiver and access issue. I will move the home visit to tomorrow afternoon."
    elif next_action == "schedule_callback":
        patient_reply = "I am sorry the pain has worsened. I am arranging a callback before tomorrow's visit."
    elif next_action == "priority_callback":
        patient_reply = "I am sorry the pain has increased. I am marking this as a priority callback so our team can contact you quickly."
    elif next_action == "escalate_for_emergency_attention":
        patient_reply = (
            "This sounds urgent. Please seek emergency attention immediately while I flag this to the care team right away."
        )
    else:
        patient_reply = "I need one more detail so I can route this safely."

    blockers = []
    if has_caregiver:
        blockers.append("caregiver unavailable")
    if has_access_issue:
        blockers.append("home access concern")
    blocker_text = ", ".join(blockers) if blockers else "no major operational blocker"
    therapist_summary = (
        f"Patient issue: {intent}. Risk: {risk_level}. Next action: {next_action}. "
        f"Operational note: {blocker_text}."
    )

    return {
        "intent": intent,
        "risk_level": risk_level,
        "next_action": next_action,
        "secondary_actions": secondary_actions,
        "patient_reply": patient_reply,
        "therapist_summary": therapist_summary,
        "risk_flag": heuristic_risk_flag(observation, risk_level),
    }


def heuristic_risk_flag(observation: dict, risk_level: str) -> str:
    if observation["task_family"] == "booking":
        return "standard_booking_case"
    if observation["task_family"] == "rescheduling":
        return "caregiver_access_blocker"
    if observation["task_family"] == "callback":
        return "worsening_pain_callback"
    if risk_level == "critical":
        return "critical_pain_case"
    return "priority_pain_case"


class HeuristicPolicy:
    def predict(self, observation: dict) -> dict:
        return heuristic_decision(observation)
