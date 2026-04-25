from __future__ import annotations

import json
from typing import get_args

from app.models import CareCoordinationOutput, NextActionName

_ALL_ACTIONS = set(get_args(NextActionName))


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


def validate_submission_payload(payload: dict, observation: dict) -> dict:
    submission = CareCoordinationOutput.model_validate(payload)
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
