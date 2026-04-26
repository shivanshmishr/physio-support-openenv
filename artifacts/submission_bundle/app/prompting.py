from __future__ import annotations

import json


SYSTEM_PROMPT = """
You are an assistant coordinating home physiotherapy support.
Return only compact JSON with the exact keys:
intent, risk_level, next_action, secondary_actions, patient_reply, therapist_summary, risk_flag.

Rules:
- next_action must be one of the case allowed_actions.
- secondary_actions must be additional useful actions, not random filler.
- Keep patient_reply empathetic, concise, and action-oriented.
- Keep therapist_summary operational and specific.
- therapist_summary must explicitly cover:
  1. patient issue,
  2. risk level,
  3. next action,
  4. blockers or logistics context,
  5. timing or scheduled visit context when available.
- patient_reply should state what happens next and include timing when the case provides one.
- Safety first: severe pain or urgent symptom change must not be handled like routine scheduling.
- If the message mixes pain escalation with logistics, prioritize the safety action first.
- Do not add markdown, code fences, explanations, or extra keys.
- Start the answer with { and end the answer with }.
""".strip()


def schema_example() -> dict:
    return {
        "intent": "mixed_intent",
        "risk_level": "high",
        "next_action": "priority_callback",
        "secondary_actions": ["notify_therapist"],
        "patient_reply": "I'm sorry the pain has increased. I'm marking this as a priority callback so our team can contact you quickly.",
        "therapist_summary": "Patient issue: worsening pain before next visit. Risk: high. Next action: priority_callback. Operational blockers: none reported. Scheduled context: callback needed before tomorrow's visit. Secondary actions: notify_therapist.",
        "risk_flag": "priority_pain_case",
    }


def build_user_prompt(observation: dict) -> str:
    policy_hint = _primary_action_hint(observation)
    return (
        "Produce one structured care-coordination decision for this case.\n"
        "Return JSON only.\n"
        "Do not include any explanation before or after the JSON object.\n"
        f"Required schema example: {json.dumps(schema_example())}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=True)}\n"
        f"Allowed actions: {json.dumps(observation.get('allowed_actions', []), ensure_ascii=True)}\n"
        "Choose the safest operational next step, then make the reply and internal summary consistent with it.\n"
        "The therapist_summary should read like an internal operational handoff, not a generic sentence.\n"
        f"Primary action policy hint: {policy_hint}"
    )


def _primary_action_hint(observation: dict) -> str:
    task_family = observation.get("task_family", "")
    if task_family == "callback":
        return (
            "Prefer schedule_callback as the main action for worsening-pain callback requests. "
            "Use priority_callback only when the symptoms are clearly urgent."
        )
    if task_family == "rescheduling":
        return (
            "If the patient requests a direct move because of a temporary caregiver or access blocker, "
            "prefer reschedule_home_visit over softer alternatives."
        )
    if task_family == "priority_pain":
        return (
            "When pain escalation and logistics appear together, use priority callback or emergency escalation as the primary action. "
            "Put visit-plan changes into secondary_actions."
        )
    return "Prefer the exact safest operational action from allowed_actions, not a softer fallback."


def build_messages(observation: dict, assistant_response: dict | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(observation)},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": json.dumps(assistant_response, ensure_ascii=True)})
    return messages


def render_plain_prompt(observation: dict) -> str:
    return (
        "### System\n"
        f"{SYSTEM_PROMPT}\n\n"
        "### User\n"
        f"{build_user_prompt(observation)}\n\n"
        "### Assistant\n"
    )


def render_plain_training_text(observation: dict, assistant_response: dict) -> str:
    return render_plain_prompt(observation) + json.dumps(assistant_response, ensure_ascii=True)
