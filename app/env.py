from __future__ import annotations

from copy import deepcopy

from app.models import Action, Observation


class PhysioSupportEnv:
    def __init__(self, task: dict):
        self.task = deepcopy(task)
        self.current_state: Observation | None = None
        self.done = False
        self.history: list[str] = []
        self.flow_progress = 0

    def reset(self) -> dict:
        self.done = False
        self.history = []
        self.flow_progress = 0
        self.current_state = Observation(
            task_id=self.task["task_id"],
            task_type=self.task["task_type"],
            patient_message=self.task["patient_message"],
            known_info=self.task["known_info"],
            available_slots=list(self.task["available_slots"]),
            conversation_stage="collect_info",
            last_action_error=None,
            steps_taken=0,
            booking_status="pending",
        )
        return self.state()

    def state(self) -> dict:
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        state = self.current_state.model_dump()
        state["flow_progress"] = self.flow_progress
        state["allowed_actions"] = self._allowed_actions()
        return state

    def _get_flow_progress(self) -> int:
        return self.flow_progress

    def _set_flow_progress(self, value: int) -> None:
        self.flow_progress = value

    def _allowed_actions(self) -> list[str]:
        if self.current_state is None:
            return []

        progress = self._get_flow_progress()

        if self.task["task_type"] == "new_booking":
            if progress == 0:
                return ["ask_pincode"]
            if progress == 1:
                return ["show_available_slots"]
            if progress == 2:
                return ["book_slot"]
            if progress == 3:
                return ["confirm_completion"]
            return []

        if self.task["task_type"] == "reschedule":
            if progress == 0:
                return ["show_available_slots"]
            if progress == 1:
                return ["reschedule_slot"]
            if progress == 2:
                return ["confirm_completion"]
            return []

        if progress == 0:
            return ["escalate_to_human"]
        if progress == 1:
            return ["confirm_completion"]
        return []

    def step(self, action_input: dict) -> tuple[dict, float, bool, dict]:
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.done:
            return self.state(), 0.0, True, {"error": "Episode already completed"}

        action = Action.model_validate(action_input)
        self.current_state.steps_taken += 1
        reward = -0.2
        reason = "Incorrect action for current state"
        error = None

        if self.current_state.steps_taken > self.task["max_steps"]:
            self.done = True
            self.current_state.booking_status = "failed"
            self.current_state.last_action_error = "Max steps exceeded"
            return self.state(), -1.0, True, {"error": "Max steps exceeded", "reason": "too_many_steps"}

        if action.action in self.history:
            reward -= 0.1

        if self.task["task_type"] == "new_booking":
            reward, reason, error = self._handle_new_booking(action)
        elif self.task["task_type"] == "reschedule":
            reward, reason, error = self._handle_reschedule(action)
        else:
            reward, reason, error = self._handle_escalation(action)

        self.history.append(action.action)
        self.current_state.last_action_error = error
        return self.state(), reward, self.done, {"error": error, "reason": reason}

    def _handle_new_booking(self, action: Action) -> tuple[float, str, str | None]:
        info = self.current_state.known_info
        progress = self._get_flow_progress()

        if action.action == "ask_pincode" and info.pincode is None and progress == 0:
            info.pincode = self.task["serviceable_pincodes"][0]
            self.current_state.conversation_stage = "select_action"
            self._set_flow_progress(1)
            return 0.16, "Collected missing pincode", None

        if action.action == "show_available_slots" and info.pincode is not None and progress == 1:
            self.current_state.conversation_stage = "select_action"
            self._set_flow_progress(2)
            return 0.24, "Displayed valid slots", None

        if action.action == "book_slot" and action.slot_id == self.task["valid_slot"] and progress == 2:
            self.current_state.booking_status = f"booked:{action.slot_id}"
            self.current_state.conversation_stage = "finalize"
            self._set_flow_progress(3)
            return 0.40, "Booked the correct slot", None

        if action.action == "confirm_completion" and str(self.current_state.booking_status).startswith("booked:") and progress == 3:
            self.done = True
            self.current_state.conversation_stage = "done"
            self._set_flow_progress(4)
            return 0.16, "Completed booking flow", None

        if action.action == "escalate_to_human":
            self.done = True
            self.current_state.booking_status = "escalated"
            self.current_state.conversation_stage = "done"
            return -1.0, "Escalated a solvable request", "Unnecessary escalation"

        return -0.2, "Action does not match booking flow order", "Invalid action for new booking task"

    def _handle_reschedule(self, action: Action) -> tuple[float, str, str | None]:
        info = self.current_state.known_info
        progress = self._get_flow_progress()

        if action.action == "show_available_slots" and info.existing_booking is not None and progress == 0:
            self.current_state.conversation_stage = "select_action"
            self._set_flow_progress(1)
            return 0.285, "Displayed reschedule options", None

        if action.action == "reschedule_slot" and action.slot_id == self.task["valid_slot"] and progress == 1:
            info.existing_booking = action.slot_id
            self.current_state.booking_status = f"rescheduled:{action.slot_id}"
            self.current_state.conversation_stage = "finalize"
            self._set_flow_progress(2)
            return 0.475, "Rescheduled to a valid slot", None

        if action.action == "confirm_completion" and str(self.current_state.booking_status).startswith("rescheduled:") and progress == 2:
            self.done = True
            self.current_state.conversation_stage = "done"
            self._set_flow_progress(3)
            return 0.19, "Completed reschedule flow", None

        if action.action == "book_slot":
            return -1.0, "Created duplicate booking instead of rescheduling", "Use reschedule_slot for this task"

        return -0.2, "Action does not match reschedule flow order", "Invalid action for reschedule task"

    def _handle_escalation(self, action: Action) -> tuple[float, str, str | None]:
        progress = self._get_flow_progress()

        if action.action == "escalate_to_human" and progress == 0:
            self.current_state.booking_status = "escalated"
            self.current_state.conversation_stage = "finalize"
            self._set_flow_progress(1)
            return 0.76, "Safely escalated urgent case", None

        if action.action == "confirm_completion" and self.current_state.booking_status == "escalated" and progress == 1:
            self.done = True
            self.current_state.conversation_stage = "done"
            self._set_flow_progress(2)
            return 0.19, "Completed escalation flow", None

        if action.action in {"book_slot", "show_available_slots", "reschedule_slot"}:
            self.done = True
            self.current_state.booking_status = "failed"
            self.current_state.conversation_stage = "done"
            return -1.0, "Unsafe handling of urgent case", "Urgent task must be escalated"

        return -0.2, "Action does not match escalation flow order", "Invalid action for escalation task"

    def close(self) -> None:
        self.done = True
