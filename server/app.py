from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.env import PhysioSupportEnv
from app.grader import grade_episode, normalize_task_score, reward_spec
from app.models import CareCoordinationAction, Observation, PhysioSupportState
from app.tasks import TASKS, TASKS_BY_ID
from inference import build_client, choose_action

app = FastAPI(title="PhysioSupportEnv", version="0.1.0")
SESSIONS: dict[str, PhysioSupportEnv] = {}


class ResetRequest(BaseModel):
    task_id: str | None = None


@app.get("/")
def root() -> dict:
    return {
        "name": "physio-support-env",
        "status": "ok",
        "endpoints": [
            "/health",
            "/tasks",
            "/metadata",
            "/schema",
            "/reward_spec",
            "/reset",
            "/state/{session_id}",
            "/step/{session_id}",
            "/run_inference/{task_id}",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return PhysioSupportEnv().get_metadata().model_dump()


@app.get("/schema")
def schema() -> dict:
    return {
        "action": CareCoordinationAction.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": PhysioSupportState.model_json_schema(),
    }


@app.get("/reward_spec")
def get_reward_spec() -> dict:
    return reward_spec()


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "patient_message": task["patient_message"],
                "max_steps": task["max_steps"],
            }
            for task in TASKS
        ]
    }


@app.post("/reset")
def reset_env(request: ResetRequest | None = None) -> dict:
    requested_task_id = request.task_id if request and request.task_id else TASKS[0]["task_id"]
    task = TASKS_BY_ID.get(requested_task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {requested_task_id}")

    session_id = str(uuid4())
    env = PhysioSupportEnv(requested_task_id)
    observation = env.reset_dict()
    SESSIONS[session_id] = env
    return {"session_id": session_id, "observation": observation, "done": False, "reward": None}


@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return {"session_id": session_id, "state": env.state_dict(), "done": env.done}


@app.post("/step/{session_id}")
def step_env(session_id: str, request: dict) -> dict:
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    action = request.get("action", request)
    observation, reward, done, info = env.step_dict(action)
    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "task_score": float(info.get("task_score", normalize_task_score(reward))),
        "raw_reward": float(info.get("raw_total_reward", reward)),
        "done": done,
        "info": info,
    }


@app.post("/run_inference/{task_id}")
def run_inference(task_id: str) -> dict:
    task = TASKS_BY_ID.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    client = build_client()
    env = PhysioSupportEnv(task_id=task_id)
    observation = env.reset_dict()
    steps: list[dict] = []
    raw_total_reward = 0.0
    done = False
    step_number = 0
    info: dict = {}

    while not done and step_number < task["max_steps"]:
        try:
            action, action_source = choose_action(client, observation)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM action generation failed: {exc}") from exc

        observation, reward, done, info = env.step_dict(action)
        step_number += 1
        raw_total_reward += float(info.get("raw_total_reward", reward))
        steps.append(
            {
                "step": step_number,
                "decision": action,
                "source": action_source,
                "reward": reward,
                "task_score": float(info.get("task_score", normalize_task_score(reward))),
                "raw_reward": float(info.get("raw_total_reward", reward)),
                "done": done,
                "info": info,
                "observation": observation,
            }
        )

    score = float(info.get("task_score", grade_episode(raw_total_reward, env.state_dict())))
    score = normalize_task_score(score)

    return {
        "task_id": task_id,
        "model": "llm" if client is not None else "heuristic",
        "score": score,
        "success": score >= float(task.get("success_score_threshold", 0.8)),
        "raw_total_reward": raw_total_reward,
        "done": done,
        "steps": steps,
        "final_state": env.state_dict(),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
