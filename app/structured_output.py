from __future__ import annotations

import json
import re
from typing import get_args

from app.models import CareCoordinationOutput, NextActionName

_ALL_ACTIONS = set(get_args(NextActionName))
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


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
