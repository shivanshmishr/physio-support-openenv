from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ActionName = Literal[
    "ask_pincode",
    "ask_time_preference",
    "show_available_slots",
    "book_slot",
    "reschedule_slot",
    "cancel_booking",
    "escalate_to_human",
    "confirm_completion",
]

TaskType = Literal["new_booking", "reschedule", "escalation"]
ConversationStage = Literal["collect_info", "select_action", "finalize", "done"]


class KnownInfo(BaseModel):
    pincode: Optional[str] = None
    preferred_time: Optional[str] = None
    existing_booking: Optional[str] = None
    urgency: str = "normal"


class Observation(BaseModel):
    task_id: str
    task_type: TaskType
    patient_message: str
    known_info: KnownInfo
    available_slots: list[str] = Field(default_factory=list)
    conversation_stage: ConversationStage
    last_action_error: Optional[str] = None
    steps_taken: int = 0
    booking_status: str = "pending"


class Action(BaseModel):
    action: ActionName
    slot_id: Optional[str] = None


class Reward(BaseModel):
    value: float
    reason: str
