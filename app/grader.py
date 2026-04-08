_EPS = 1e-6


def _open_unit_interval(value: float) -> float:
    """Clamp score to the open interval (0, 1)."""
    if value <= _EPS:
        return _EPS
    if value >= 1.0 - _EPS:
        return 1.0 - _EPS
    return value


def grade_episode(total_reward: float, final_state: dict) -> float:
    status = str(final_state.get("booking_status", ""))
    stage = final_state.get("conversation_stage")

    if stage != "done":
        return _open_unit_interval(0.0)

    if status.startswith("booked:") or status.startswith("rescheduled:") or status == "escalated":
        if total_reward >= 0.9:
            return _open_unit_interval(1.0)
        if total_reward >= 0.5:
            return _open_unit_interval(0.7)
        return _open_unit_interval(0.3)

    return _open_unit_interval(0.0)
