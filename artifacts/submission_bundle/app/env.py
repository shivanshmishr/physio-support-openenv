from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from app.grader import grade_episode, score_submission
from app.models import CareCoordinationAction, Observation, PhysioSupportState
from app.tasks import TASKS, TASKS_BY_ID


class PhysioSupportEnv(Environment[CareCoordinationAction, Observation, PhysioSupportState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str | None = None, task: Optional[dict] = None):
        super().__init__()
        if task is not None:
            self.task = deepcopy(task)
        else:
            self.task = deepcopy(TASKS_BY_ID[task_id or TASKS[0]["task_id"]])
        self._validate_task_spec(self.task)
        self.current_state: Optional[PhysioSupportState] = None
        self.done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if task_id is not None:
            if task_id not in TASKS_BY_ID:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.task = deepcopy(TASKS_BY_ID[task_id])
            self._validate_task_spec(self.task)

        self.done = False
        self.current_state = PhysioSupportState(
            episode_id=episode_id or self.task["task_id"],
            step_count=0,
            task_id=self.task["task_id"],
            task_family=self.task["task_family"],
            patient_message=self.task["patient_message"],
            patient_history_summary=self.task["patient_history_summary"],
            care_plan_summary=self.task["care_plan_summary"],
            appointment_context=deepcopy(self.task["appointment_context"]),
            visit_context=deepcopy(self.task["visit_context"]),
            operational_constraints=list(self.task["operational_constraints"]),
            allowed_actions=list(self.task["allowed_actions"]),
            policy_constraints=list(self.task["policy_constraints"]),
            conversation_history=[],
            episode_status="pending",
            last_action_error=None,
            last_submission=None,
            last_reward=None,
            last_raw_reward=None,
            last_reward_breakdown={},
            penalties=[],
            unsafe=False,
            hidden_state=deepcopy(self.task["hidden_state"]),
            max_steps=self.task["max_steps"],
        )
        self._reset_rubric()
        return self._build_observation()

    def step(
        self,
        action: CareCoordinationAction | dict,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.done:
            return self._build_observation(
                reward=0.0,
                done=True,
                metadata={"error": "Episode already completed", "reason": "done"},
            )

        self.current_state.step_count += 1
        validation_error = None
        submission = None

        try:
            submission = action if isinstance(action, CareCoordinationAction) else CareCoordinationAction.model_validate(action)
        except Exception as exc:
            validation_error = str(exc)

        reward_result = score_submission(submission, self.task, validation_error)
        raw_total_reward = reward_result.total_reward

        raw_submission = action.model_dump(exclude_none=True) if isinstance(action, CareCoordinationAction) else deepcopy(action)
        self.current_state.last_submission = raw_submission
        self.current_state.last_action_error = validation_error
        self.current_state.last_raw_reward = raw_total_reward
        self.current_state.last_reward_breakdown = reward_result.component_scores
        self.current_state.penalties = reward_result.penalties
        self.current_state.unsafe = reward_result.unsafe
        self.current_state.episode_status = "completed" if reward_result.passed else "failed"

        task_score = grade_episode(
            raw_total_reward,
            {
                "episode_status": self.current_state.episode_status,
                "unsafe": self.current_state.unsafe,
            },
        )
        self.current_state.last_reward = task_score

        if submission is not None:
            self.current_state.conversation_history.append(submission.patient_reply)

        self.done = True

        return self._build_observation(
            reward=task_score,
            done=self.done,
            metadata={
                "error": validation_error,
                "reason": reward_result.reason,
                "task_score": task_score,
                "raw_total_reward": raw_total_reward,
                "breakdown": reward_result.component_scores,
                "penalties": reward_result.penalties,
                "penalty_values": reward_result.penalty_values,
                "passed": reward_result.passed,
                "unsafe": reward_result.unsafe,
            },
        )

    @property
    def state(self) -> PhysioSupportState:
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.current_state

    def state_dict(self) -> dict:
        return self.state.model_dump()

    def reset_dict(self, task_id: Optional[str] = None) -> dict:
        return self.reset(task_id=task_id).model_dump()

    def step_dict(self, action_input: dict) -> tuple[dict, float, bool, dict]:
        observation = self.step(action_input)
        metadata = dict(observation.metadata)
        reward = float(observation.reward) if observation.reward is not None else 0.0
        return observation.model_dump(), reward, observation.done, metadata

    def close(self) -> None:
        self.done = True

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="physio-support-env",
            description="OpenEnv environment for home physiotherapy care coordination with safety-sensitive workflow decisions.",
            version="0.1.0",
            author="MetaHackathon team",
        )

    def _build_observation(
        self,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Observation:
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return Observation(
            task_id=self.current_state.task_id,
            task_family=self.current_state.task_family,
            patient_message=self.current_state.patient_message,
            patient_history_summary=self.current_state.patient_history_summary,
            care_plan_summary=self.current_state.care_plan_summary,
            appointment_context=deepcopy(self.current_state.appointment_context),
            recent_history=list(self.current_state.conversation_history[-3:]),
            visit_context=deepcopy(self.current_state.visit_context),
            operational_constraints=list(self.current_state.operational_constraints),
            allowed_actions=list(self.current_state.allowed_actions),
            policy_constraints=list(self.current_state.policy_constraints),
            step_id=self.current_state.step_count,
            max_steps=self.current_state.max_steps,
            done=self.done if done is None else done,
            reward=reward,
            metadata=metadata or {},
        )

    def _validate_task_spec(self, task: dict) -> None:
        required_keys = {
            "task_id",
            "task_family",
            "patient_message",
            "patient_history_summary",
            "care_plan_summary",
            "appointment_context",
            "visit_context",
            "operational_constraints",
            "allowed_actions",
            "policy_constraints",
            "hidden_state",
            "max_steps",
            "truth",
        }
        missing = sorted(required_keys - set(task.keys()))
        if missing:
            raise ValueError(f"Task '{task.get('task_id', 'unknown')}' missing required keys: {missing}")
