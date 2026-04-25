from __future__ import annotations

import json
import os

from openai import OpenAI

from app.env import PhysioSupportEnv
from app.grader import grade_episode, normalize_task_score
from app.heuristic_policy import heuristic_decision
from app.model_policy import load_model_policy_from_env
from app.prompting import SYSTEM_PROMPT, build_user_prompt, schema_example
from app.structured_output import extract_json_object, validate_submission_payload
from app.tasks import TASKS

def build_client() -> OpenAI | None:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def llm_action(client: OpenAI, observation: dict) -> dict:
    model = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    user_prompt = build_user_prompt(observation)

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
            return validate_submission_payload(action, observation)
        except ValueError as exc:
            if attempt == 1:
                raise
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was invalid. "
                        f"{exc}. Return corrected JSON only and keep next_action inside allowed_actions. "
                        f"Schema example: {json.dumps(schema_example())}"
                    ),
                }
            )

    raise RuntimeError("LLM action generation failed after retries")


def choose_action(client: OpenAI | None, observation: dict) -> tuple[dict, str]:
    local_model_policy = load_model_policy_from_env()
    if local_model_policy is not None:
        return local_model_policy.predict(observation), "local_model"

    if client is not None:
        try:
            return llm_action(client, observation), "llm"
        except Exception:
            pass

    return heuristic_decision(observation), "heuristic"


def format_action_for_log(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), sort_keys=True)


def main() -> None:
    client = build_client()
    model_name = os.getenv("LOCAL_ADAPTER_PATH") or os.getenv("LOCAL_BASE_MODEL") or os.getenv("MODEL_NAME", "heuristic")

    for task in TASKS:
        env = PhysioSupportEnv(task["task_id"])
        observation = env.reset_dict()
        raw_total_reward = 0.0
        rewards: list[float] = []
        step_number = 0
        done = False
        success = False
        score = 0.0
        info: dict = {}

        print(f"[START] task={task['task_id']} env=physio_support model={model_name}")
        try:
            while not done and step_number < task["max_steps"]:
                action, source = choose_action(client, observation)
                observation, reward, done, info = env.step_dict(action)
                step_number += 1
                raw_total_reward += float(info.get("raw_total_reward", reward))
                rewards.append(reward)
                error_text = info["error"] if info["error"] else "null"
                print(
                    f"[STEP] step={step_number} source={source} action={format_action_for_log(action)} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
                )

            score = float(info.get("task_score", grade_episode(raw_total_reward, env.state_dict())))
            score = normalize_task_score(score)
            success = score >= float(task.get("success_score_threshold", 0.72))
        finally:
            try:
                env.close()
            except Exception:
                pass
            rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
            print(
                "[END] "
                f"success={str(success).lower()} "
                f"score={score:.6f} "
                f"raw_total_reward={raw_total_reward:.6f} "
                f"steps={step_number} "
                f"rewards={rewards_str}"
            )


if __name__ == "__main__":
    main()
