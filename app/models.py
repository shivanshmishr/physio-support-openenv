from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server import Action as OpenEnvAction
from openenv.core.env_server import Observation as OpenEnvObservation
from openenv.core.env_server import State as OpenEnvState


IntentName = Literal[
    "book_visit",
    "reschedule_visit",
    "cancel_visit",
    "request_callback",
    "report_worsening_pain",
    "ask_general_info",
    "caregiver_unavailable",
    "home_access_issue",
    "mixed_intent",
]

RiskLevel = Literal["low", "medium", "high", "critical"]

NextActionName = Literal[
    "confirm_home_visit",
    "reschedule_home_visit",
    "request_more_information",
    "schedule_callback",
    "priority_callback",
    "notify_therapist",
    "convert_to_remote_checkin",
    "modify_visit_plan",
    "escalate_for_clinical_review",
    "escalate_for_emergency_attention",
    "close_with_guidance",
]

TaskFamily = Literal["booking", "rescheduling", "callback", "priority_pain"]
EpisodeStatus = Literal["pending", "completed", "failed"]


class CareCoordinationAction(OpenEnvAction):
    intent: IntentName
    risk_level: RiskLevel
    next_action: NextActionName
    secondary_actions: list[NextActionName] = Field(default_factory=list)
    patient_reply: str = Field(min_length=12, max_length=400)
    therapist_summary: str = Field(min_length=12, max_length=500)
    risk_flag: Optional[str] = None


CareCoordinationOutput = CareCoordinationAction


class Observation(OpenEnvObservation):
    task_id: str
    task_family: TaskFamily
    patient_message: str
    patient_history_summary: str
    care_plan_summary: str
    appointment_context: dict[str, Any]
    recent_history: list[str] = Field(default_factory=list)
    visit_context: dict[str, Any]
    operational_constraints: list[str] = Field(default_factory=list)
    allowed_actions: list[NextActionName] = Field(default_factory=list)
    policy_constraints: list[str] = Field(default_factory=list)
    step_id: int = 0
    max_steps: int = 1


class HiddenState(BaseModel):
    true_clinical_severity: RiskLevel
    home_visit_suitability: Literal["suitable", "needs_review", "unsuitable"]
    latent_adherence_risk: Literal["low", "medium", "high"]
    escalation_necessity: Literal["none", "callback", "priority_callback", "emergency"]
    serviceability_status: Literal["serviceable", "restricted", "blocked"]


class PhysioSupportState(OpenEnvState):
    task_id: str
    task_family: TaskFamily
    patient_message: str
    patient_history_summary: str
    care_plan_summary: str
    appointment_context: dict[str, Any]
    visit_context: dict[str, Any]
    operational_constraints: list[str] = Field(default_factory=list)
    allowed_actions: list[NextActionName] = Field(default_factory=list)
    policy_constraints: list[str] = Field(default_factory=list)
    conversation_history: list[str] = Field(default_factory=list)
    episode_status: EpisodeStatus = "pending"
    last_action_error: Optional[str] = None
    last_submission: Optional[dict[str, Any]] = None
    last_reward: Optional[float] = None
    last_raw_reward: Optional[float] = None
    last_reward_breakdown: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)
    unsafe: bool = False
    hidden_state: HiddenState
    max_steps: int = 1


class RewardResult(BaseModel):
    total_reward: float
    component_scores: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)
    penalty_values: dict[str, float] = Field(default_factory=dict)
    reason: str
    passed: bool
    unsafe: bool = False
