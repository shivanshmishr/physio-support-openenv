from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.env import PhysioSupportEnv
from app.tasks import TASKS
from inference import build_client, choose_action


app = FastAPI(title="PhysioSupportEnv", version="0.1.0")

TASKS_BY_ID = {task["task_id"]: task for task in TASKS}
SESSIONS: dict[str, PhysioSupportEnv] = {}


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: str
    slot_id: str | None = None


@app.get("/")
def root() -> dict:
    return {
        "name": "physio-support-env",
        "status": "ok",
        "endpoints": ["/health", "/tasks", "/reset", "/state/{session_id}", "/step/{session_id}"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": [
            {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
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
    env = PhysioSupportEnv(task)
    state = env.reset()
    SESSIONS[session_id] = env
    return {"session_id": session_id, "state": state}


@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return {"session_id": session_id, "state": env.state(), "done": env.done}


@app.post("/step/{session_id}")
def step_env(session_id: str, request: StepRequest) -> dict:
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    state, reward, done, info = env.step(request.model_dump(exclude_none=True))
    return {
        "session_id": session_id,
        "state": state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/run_inference/{task_id}")
def run_inference(task_id: str) -> dict:
    task = TASKS_BY_ID.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    client = build_client()
    if client is None:
        raise HTTPException(status_code=500, detail="No API client configured. Set HF_TOKEN or OPENAI_API_KEY.")

    env = PhysioSupportEnv(task)
    observation = env.reset()
    steps: list[dict] = []
    total_reward = 0.0
    done = False
    step_number = 0

    while not done and step_number < task["max_steps"]:
        try:
            action, action_source = choose_action(client, observation)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM action generation failed: {exc}") from exc

        observation, reward, done, info = env.step(action)
        step_number += 1
        total_reward += reward
        steps.append(
            {
                "step": step_number,
                "action": action,
                "source": action_source,
                "reward": reward,
                "done": done,
                "info": info,
                "state": observation,
            }
        )

    return {
        "task_id": task_id,
        "model": "llm",
        "total_reward": total_reward,
        "done": done,
        "steps": steps,
        "final_state": observation,
    }


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
