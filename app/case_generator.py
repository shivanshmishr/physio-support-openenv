from __future__ import annotations

from copy import deepcopy


_TRAIN_PREFIXES = [
    "Hi team,",
    "Hello,",
    "Please help,",
    "I need support:",
]
_TRAIN_SUFFIXES = [
    "Please advise on the safest next step.",
    "Kindly guide me on what happens next.",
    "I want to make sure this is handled correctly.",
    "Please let me know the next action.",
]
_EVAL_PREFIXES = [
    "Quick update,",
    "Can you assist?",
    "Need urgent coordination:",
    "Checking in because this changed:",
]
_EVAL_SUFFIXES = [
    "I need support coordination today.",
    "Please respond with the right plan.",
    "Please confirm the safest operational next step.",
    "I want the team to handle this correctly.",
]
_SUMMARY_SUFFIXES = {
    "history": [
        "",
        "Additional history context remains stable.",
        "Current history note remains unchanged.",
        "Recent history still supports the same routing decision.",
    ],
    "plan": [
        "",
        "Additional plan context remains stable.",
        "Current care-plan note remains unchanged.",
        "Existing plan details still apply for routing.",
    ],
}
_TRAIN_CONSTRAINT_NOTES = [
    "Coordinator should keep the reply concise and operationally clear.",
    "Operational summary should name the blocker explicitly if one exists.",
]
_EVAL_CONSTRAINT_NOTES = [
    "Do not confuse patient reassurance with visit confirmation.",
    "Held-out case phrasing may differ while the underlying policy stays the same.",
]


def build_case_splits(base_tasks: list[dict], variants_per_task: int = 6) -> tuple[list[dict], list[dict]]:
    if variants_per_task < 2:
        raise ValueError("variants_per_task must be at least 2 so both train and eval splits exist.")

    train_count = max(1, int(variants_per_task * 0.67))
    eval_count = max(1, variants_per_task - train_count)

    train_cases: list[dict] = []
    eval_cases: list[dict] = []

    for task in base_tasks:
        train_cases.extend(expand_task_variants(task, train_count, split="train"))
        eval_cases.extend(expand_task_variants(task, eval_count, split="eval"))

    validate_case_splits(train_cases, eval_cases, expected_base_task_ids={task["task_id"] for task in base_tasks})
    return train_cases, eval_cases


def expand_task_variants(task: dict, variant_count: int, split: str) -> list[dict]:
    if split not in {"train", "eval"}:
        raise ValueError(f"Unsupported split: {split}")

    variants: list[dict] = []
    prefixes = _TRAIN_PREFIXES if split == "train" else _EVAL_PREFIXES
    suffixes = _TRAIN_SUFFIXES if split == "train" else _EVAL_SUFFIXES

    for variant_index in range(variant_count):
        variant = deepcopy(task)
        variant_number = variant_index + 1
        variant["base_task_id"] = task["task_id"]
        variant["split"] = split
        variant["variant_id"] = f"{split}_v{variant_number}"
        variant["task_id"] = f"{task['task_id']}_{variant['variant_id']}"
        variant["patient_message"] = _vary_message(task["patient_message"], variant_index, prefixes, suffixes)
        variant["patient_history_summary"] = _vary_summary(task["patient_history_summary"], variant_index, "history")
        variant["care_plan_summary"] = _vary_summary(task["care_plan_summary"], variant_index, "plan")
        variant["operational_constraints"] = _vary_constraints(task["operational_constraints"], variant_index, split)
        variant["policy_constraints"] = _vary_policy_constraints(task["policy_constraints"], variant_index, split)
        variant["variant_profile"] = {
            "split": split,
            "variant_index": variant_number,
            "message_style": "prefixed" if variant_index % 2 == 0 else "plain",
            "summary_style": variant_index % 4,
            "constraint_style": "held_out_note" if split == "eval" else "train_note",
        }
        variants.append(variant)

    return variants


def validate_case_splits(
    train_cases: list[dict],
    eval_cases: list[dict],
    expected_base_task_ids: set[str] | None = None,
) -> None:
    train_ids = {case["task_id"] for case in train_cases}
    eval_ids = {case["task_id"] for case in eval_cases}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"Train/eval case overlap detected: {overlap[:5]}")

    train_base_ids = {case["base_task_id"] for case in train_cases}
    eval_base_ids = {case["base_task_id"] for case in eval_cases}
    missing_eval = sorted(train_base_ids - eval_base_ids)
    missing_train = sorted(eval_base_ids - train_base_ids)
    if missing_eval or missing_train:
        raise ValueError(
            "Every base task must appear in both train and eval splits. "
            f"Missing eval={missing_eval}, missing train={missing_train}"
        )

    if expected_base_task_ids is not None:
        missing_expected = sorted(expected_base_task_ids - train_base_ids - eval_base_ids)
        if missing_expected:
            raise ValueError(f"Expected base tasks missing from generated splits: {missing_expected}")

    for case in train_cases:
        if case.get("split") != "train":
            raise ValueError(f"Train case has incorrect split marker: {case['task_id']}")
    for case in eval_cases:
        if case.get("split") != "eval":
            raise ValueError(f"Eval case has incorrect split marker: {case['task_id']}")


def case_split_summary(train_cases: list[dict], eval_cases: list[dict]) -> dict:
    families = sorted({case["task_family"] for case in train_cases + eval_cases})
    base_task_ids = sorted({case["base_task_id"] for case in train_cases + eval_cases})
    return {
        "base_task_count": len(base_task_ids),
        "train_case_count": len(train_cases),
        "eval_case_count": len(eval_cases),
        "task_families": families,
        "train_by_family": {family: sum(1 for case in train_cases if case["task_family"] == family) for family in families},
        "eval_by_family": {family: sum(1 for case in eval_cases if case["task_family"] == family) for family in families},
        "train_base_task_ids": sorted({case["base_task_id"] for case in train_cases}),
        "eval_base_task_ids": sorted({case["base_task_id"] for case in eval_cases}),
    }


def _vary_message(message: str, variant_index: int, prefixes: list[str], suffixes: list[str]) -> str:
    prefix = prefixes[variant_index % len(prefixes)]
    suffix = suffixes[variant_index % len(suffixes)]
    if variant_index % 2 == 0:
        return f"{prefix} {message} {suffix}"
    return f"{message} {suffix}"


def _vary_summary(summary: str, variant_index: int, label: str) -> str:
    suffix = _SUMMARY_SUFFIXES[label][variant_index % len(_SUMMARY_SUFFIXES[label])]
    return f"{summary} {suffix}".strip()


def _vary_constraints(constraints: list[str], variant_index: int, split: str) -> list[str]:
    notes = _TRAIN_CONSTRAINT_NOTES if split == "train" else _EVAL_CONSTRAINT_NOTES
    varied = list(constraints)
    note = notes[variant_index % len(notes)]
    if note not in varied:
        varied.append(note)
    if split == "eval" and variant_index % 2 == 1:
        return list(reversed(varied))
    return varied


def _vary_policy_constraints(policy_constraints: list[str], variant_index: int, split: str) -> list[str]:
    varied = list(policy_constraints)
    if split == "eval":
        held_out_note = "Apply the same safety policy even when the wording differs from training phrasing."
        if held_out_note not in varied:
            varied.append(held_out_note)
    if variant_index % 2 == 1:
        return list(reversed(varied))
    return varied
