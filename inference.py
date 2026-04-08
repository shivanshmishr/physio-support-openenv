from __future__ import annotations

import json
import os

from openai import OpenAI

from app.env import PhysioSupportEnv
from app.tasks import TASKS


SYSTEM_PROMPT = """
You are an assistant choosing the next action for a physiotherapy support workflow.
Return only compact JSON with keys: action and optional slot_id.
Allowed actions: ask_pincode, ask_time_preference, show_available_slots, book_slot, reschedule_slot, cancel_booking, escalate_to_human, confirm_completion.
Choose the safest next action based on the state.
Rules:
- If action is book_slot or reschedule_slot, slot_id must exactly match one value from available_slots.
- For all other actions, do not include slot_id.
- Do not invent placeholders like "time", "new_slot", "pincode", or "task_type".
- Prefer the next sensible workflow action instead of repeating the same invalid action.
- You must choose the action from allowed_actions.
- Follow workflow order strictly using task_type and flow_progress.
- For new_booking:
  flow_progress 0 -> ask_pincode if pincode is missing
  flow_progress 1 -> show_available_slots
  flow_progress 2 -> book_slot with the valid slot from available_slots
  flow_progress 3 -> confirm_completion
- For reschedule:
  flow_progress 0 -> show_available_slots
  flow_progress 1 -> reschedule_slot with a slot from available_slots
  flow_progress 2 -> confirm_completion
- For escalation:
  flow_progress 0 -> escalate_to_human
  flow_progress 1 -> confirm_completion
""".strip()

_EPS = 1e-6


def open_unit_interval(value: float) -> float:
    """Clamp score to the open interval (0, 1)."""
    if value <= _EPS:
        return _EPS
    if value >= 1.0 - _EPS:
        return 1.0 - _EPS
    return value


def extract_json_object(content: str) -> dict:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model response did not contain a JSON object: {content}")

    return json.loads(cleaned[start : end + 1])


def validate_action_payload(action: dict, observation: dict) -> dict:
    allowed_actions = {
        "ask_pincode",
        "ask_time_preference",
        "show_available_slots",
        "book_slot",
        "reschedule_slot",
        "cancel_booking",
        "escalate_to_human",
        "confirm_completion",
    }
    action_name = action.get("action")
    available_slots = observation.get("available_slots", [])

    if action_name not in allowed_actions:
        raise ValueError(f"Invalid action returned by model: {action_name}")

    if action_name in {"book_slot", "reschedule_slot"}:
        slot_id = action.get("slot_id")
        if slot_id not in available_slots:
            raise ValueError(f"Invalid slot_id returned by model: {slot_id}. Allowed slots: {available_slots}")
        return {"action": action_name, "slot_id": slot_id}

    return {"action": action_name}


def fallback_action(observation: dict) -> dict:
    allowed_actions = observation.get("allowed_actions", [])
    if not allowed_actions:
        return {"action": "confirm_completion"}

    action_name = allowed_actions[0]
    if action_name in {"book_slot", "reschedule_slot"}:
        available_slots = observation.get("available_slots", [])
        if not available_slots:
            return {"action": "confirm_completion"}
        return {"action": action_name, "slot_id": available_slots[0]}

    return {"action": action_name}


def build_client() -> OpenAI | None:
    api_key = os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def llm_action(client: OpenAI, observation: dict) -> dict:
    model = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    user_prompt = (
        "Choose the next action for this state.\n"
        "Return JSON only.\n"
        "Examples:\n"
        'State: {"task_type":"new_booking","flow_progress":0,"known_info":{"pincode":null},"available_slots":["2026-04-05 10:00"]}\n'
        'Response: {"action":"ask_pincode"}\n'
        'State: {"task_type":"new_booking","flow_progress":2,"known_info":{"pincode":"400053"},"available_slots":["2026-04-05 10:00"]}\n'
        'Response: {"action":"book_slot","slot_id":"2026-04-05 10:00"}\n'
        'State: {"task_type":"reschedule","flow_progress":0,"available_slots":["2026-04-05 09:30"]}\n'
        'Response: {"action":"show_available_slots"}\n'
        'State: {"task_type":"reschedule","flow_progress":1,"available_slots":["2026-04-05 09:30","2026-04-05 10:30"],"allowed_actions":["reschedule_slot"]}\n'
        'Response: {"action":"reschedule_slot","slot_id":"2026-04-05 09:30"}\n'
        'State: {"task_type":"escalation","flow_progress":0,"available_slots":[]}\n'
        'Response: {"action":"escalate_to_human"}\n'
        f"State: {json.dumps(observation)}\n"
        f"Allowed actions: {json.dumps(observation.get('allowed_actions', []))}\n"
        f"Available slots you may use exactly as written: {json.dumps(observation.get('available_slots', []))}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        action = extract_json_object(content)

        try:
            return validate_action_payload(action, observation)
        except ValueError as exc:
            if attempt == 1:
                raise
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was invalid. "
                        f"{exc}. Return corrected JSON only. "
                        "If using book_slot or reschedule_slot, slot_id must be an exact entry from available_slots."
                    ),
                }
            )

    raise RuntimeError("LLM action generation failed after retries")


def choose_action(client: OpenAI | None, observation: dict) -> tuple[dict, str]:
    if client is None:
        return fallback_action(observation), "fallback"

    try:
        return llm_action(client, observation), "llm"
    except Exception as exc:
        return fallback_action(observation), "fallback"


def format_action_for_log(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), sort_keys=True)


def main() -> None:
    client = build_client()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    for task in TASKS:
        env = PhysioSupportEnv(task)
        observation = env.reset()
        total_reward = 0.0
        rewards: list[float] = []
        step_number = 0
        done = False
        success = False

        print(f"[START] task={task['task_id']} env=physio_support model={model_name}")
        try:
            while not done and step_number < task["max_steps"]:
                action, _ = choose_action(client, observation)
                observation, reward, done, info = env.step(action)
                step_number += 1
                total_reward += reward
                rewards.append(reward)
                error_text = info["error"] if info["error"] else "null"
                print(
                    f"[STEP] step={step_number} action={format_action_for_log(action)} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
                )

            max_total_reward = float(task.get("max_total_reward", 1.0))
            score = total_reward / max_total_reward if max_total_reward > 0 else 0.0
            score = open_unit_interval(score)
            success = score >= float(task.get("success_score_threshold", 0.8))
        finally:
            try:
                env.close()
            except Exception:
                pass
            rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
            print(f"[END] success={str(success).lower()} steps={step_number} rewards={rewards_str}")


if __name__ == "__main__":
    main()
