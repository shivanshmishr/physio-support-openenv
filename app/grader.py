def grade_episode(total_reward: float, final_state: dict) -> float:
    status = str(final_state.get("booking_status", ""))
    stage = final_state.get("conversation_stage")

    if stage != "done":
        return 0.0

    if status.startswith("booked:") or status.startswith("rescheduled:") or status == "escalated":
        if total_reward >= 0.9:
            return 1.0
        if total_reward >= 0.5:
            return 0.7
        return 0.3

    return 0.0
