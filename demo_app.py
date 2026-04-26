from __future__ import annotations

import html
import json
import os
from pathlib import Path

import gradio as gr

from app.env import PhysioSupportEnv
from app.heuristic_policy import heuristic_decision
from app.tasks import TASKS, TASKS_BY_ID
from inference import build_client, choose_action

FINAL_RESULTS_DIR = Path("artifacts/phase6/final_results")
TRAINING_SUMMARY_PATH = FINAL_RESULTS_DIR / "training_summary.json"
REWARD_CHART_PATH = FINAL_RESULTS_DIR / "reward_comparison.svg"
SCORE_CHART_PATH = FINAL_RESULTS_DIR / "score_comparison.svg"


def _load_training_summary() -> dict:
    if TRAINING_SUMMARY_PATH.exists():
        return json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


TRAINING_SUMMARY = _load_training_summary()


def _task_choices() -> list[tuple[str, str]]:
    return [(f"{task['task_family']} · {task['task_id']}", task["task_id"]) for task in TASKS]


def _pretty_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _render_case_markdown(task: dict) -> str:
    visit = task["visit_context"]
    appointment = task["appointment_context"]
    constraints = "\n".join(f"- {item}" for item in task["operational_constraints"])
    policies = "\n".join(f"- {item}" for item in task["policy_constraints"])
    actions = ", ".join(task["allowed_actions"])
    return (
        f"### {task['task_id']}\n\n"
        f"**Patient message**\n{task['patient_message']}\n\n"
        f"**Patient history**\n{task['patient_history_summary']}\n\n"
        f"**Care plan**\n{task['care_plan_summary']}\n\n"
        f"**Visit context**\n"
        f"- Location: {visit['location']}\n"
        f"- Preferred window: {visit['preferred_window']}\n"
        f"- Scheduled visit: {visit['scheduled_visit'] or 'None'}\n"
        f"- Caregiver required: {visit['caregiver_required']}\n"
        f"- Home access: {visit['home_access']}\n\n"
        f"**Appointment context**\n"
        f"- Existing visit: {appointment['existing_visit'] or 'None'}\n"
        f"- Therapist availability: {appointment['therapist_availability']}\n"
        f"- Serviceability notes: {appointment['serviceability_notes']}\n\n"
        f"**Allowed actions**\n`{actions}`\n\n"
        f"**Operational constraints**\n{constraints}\n\n"
        f"**Policy constraints**\n{policies}"
    )


def _render_truth_markdown(task: dict) -> str:
    truth = task["truth"]
    return (
        "### Ground Truth\n\n"
        f"- Intent: `{truth['intent']}`\n"
        f"- Risk level: `{truth['risk_level']}`\n"
        f"- Next action: `{truth['next_action']}`\n"
        f"- Risk flag: `{truth['risk_flag']}`"
    )


def load_case(task_id: str) -> tuple[str, str]:
    task = TASKS_BY_ID[task_id]
    return _render_case_markdown(task), _render_truth_markdown(task)


def _run_case(task_id: str, use_configured_policy: bool) -> tuple[str, str, str, str]:
    env = PhysioSupportEnv(task_id=task_id)
    observation = env.reset_dict()

    if use_configured_policy:
        client = build_client()
        action, source = choose_action(client, observation)
    else:
        action = heuristic_decision(observation)
        source = "heuristic"

    _, reward, _, info = env.step_dict(action)

    task_score = float(info.get("task_score", reward))
    raw_reward = float(info.get("raw_total_reward", reward))
    unsafe = bool(info.get("unsafe", False))
    penalties = info.get("penalties", [])
    breakdown = info.get("breakdown", {})

    summary_md = (
        f"### Decision Summary\n\n"
        f"- Source: `{source}`\n"
        f"- Task score: `{task_score:.3f}`\n"
        f"- Raw reward: `{raw_reward:.3f}`\n"
        f"- Unsafe: `{unsafe}`\n"
        f"- Passed: `{bool(info.get('passed', False))}`\n"
        f"- Reason: {info.get('reason', 'Scored structured care coordination output.')}"
    )

    if penalties:
        penalties_md = "### Penalties\n\n" + "\n".join(f"- {item}" for item in penalties)
    else:
        penalties_md = "### Penalties\n\n- None"

    breakdown_json = _pretty_json(breakdown)
    action_json = _pretty_json(action)
    return action_json, breakdown_json, summary_md, penalties_md


def run_heuristic(task_id: str) -> tuple[str, str, str, str]:
    return _run_case(task_id, use_configured_policy=False)


def run_configured(task_id: str) -> tuple[str, str, str, str]:
    return _run_case(task_id, use_configured_policy=True)


def _render_metrics_table() -> str:
    if not TRAINING_SUMMARY:
        return "<p>Final metrics not found.</p>"

    baseline = TRAINING_SUMMARY["baseline_metrics"]
    trained = TRAINING_SUMMARY["trained_metrics"]
    heuristic = TRAINING_SUMMARY["heuristic_metrics"]

    rows = [
        ("Baseline", baseline),
        ("Trained", trained),
        ("Heuristic", heuristic),
    ]
    header = (
        "<tr>"
        "<th>Policy</th><th>Avg Reward</th><th>Avg Score</th><th>Risk Acc</th>"
        "<th>Action Acc</th><th>Priority Recall</th><th>Unsafe Rate</th>"
        "</tr>"
    )
    body = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{metrics['avg_reward']:.4f}</td>"
        f"<td>{metrics['avg_score']:.4f}</td>"
        f"<td>{metrics['risk_accuracy']:.2f}</td>"
        f"<td>{metrics['action_accuracy']:.2f}</td>"
        f"<td>{metrics['priority_pain_recall']:.2f}</td>"
        f"<td>{metrics['unsafe_action_rate']:.2f}</td>"
        "</tr>"
        for name, metrics in rows
    )
    return f"<table class='metrics-table'>{header}{body}</table>"


def _render_svg(path: Path) -> str:
    if not path.exists():
        return "<p>Chart not available.</p>"
    return path.read_text(encoding="utf-8")


def _benchmark_markdown() -> str:
    if not TRAINING_SUMMARY:
        return "Final benchmark summary not available."

    improvement = TRAINING_SUMMARY["improvement"]
    return (
        "### Phase 6 Outcome\n\n"
        "Teacher-distilled fine-tuning improved the held-out evaluation relative to the base model.\n\n"
        f"- Average reward: `{improvement['avg_reward']:+.4f}`\n"
        f"- Average score: `{improvement['avg_score']:+.4f}`\n"
        f"- Risk accuracy: `{improvement['risk_accuracy']:+.2f}`\n"
        f"- Action accuracy: `{improvement['action_accuracy']:+.2f}`\n"
        f"- Priority pain recall: `{improvement['priority_pain_recall']:+.2f}`\n"
        "- Unsafe action rate stayed at `0.00`"
    )


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="amber",
        neutral_hue="slate",
    )
    css = """
    .app-shell {max-width: 1180px; margin: 0 auto;}
    .hero {padding: 18px 20px; border-radius: 18px; background: linear-gradient(135deg, #082f49, #0f766e 58%, #f59e0b);}
    .hero h1, .hero p {color: white !important; margin: 0;}
    .hero p {margin-top: 8px; opacity: 0.94;}
    .metrics-table {width: 100%; border-collapse: collapse; margin-top: 8px;}
    .metrics-table th, .metrics-table td {border: 1px solid #d6dee8; padding: 10px 12px; text-align: center;}
    .metrics-table th {background: #eff6ff; color: #0f172a;}
    .metrics-table td:first-child, .metrics-table th:first-child {text-align: left;}
    """

    with gr.Blocks(theme=theme, css=css, title="Physio Support OpenEnv Demo") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="hero">
                  <h1>Physio Support OpenEnv</h1>
                  <p>Judge-facing demo for structured home physiotherapy care-coordination. Inspect a seeded patient case, run a policy, and see the reward breakdown that drives evaluation.</p>
                </div>
                """
            )

            with gr.Tab("Interactive Case Runner"):
                initial_case_md, initial_truth_md = load_case(TASKS[0]["task_id"])
                with gr.Row():
                    task_id = gr.Dropdown(
                        choices=_task_choices(),
                        value=TASKS[0]["task_id"],
                        label="Seeded case",
                        info="Each case is a one-step patient support episode.",
                    )
                    load_btn = gr.Button("Load Case", variant="secondary")
                    run_heuristic_btn = gr.Button("Run Heuristic Teacher", variant="primary")
                    run_model_btn = gr.Button("Run Configured Model", variant="secondary")

                with gr.Row():
                    case_md = gr.Markdown(value=initial_case_md)
                    truth_md = gr.Markdown(value=initial_truth_md)

                with gr.Row():
                    decision_json = gr.Code(label="Model Decision JSON", language="json")
                    breakdown_json = gr.Code(label="Reward Breakdown", language="json")

                with gr.Row():
                    summary_md = gr.Markdown()
                    penalties_md = gr.Markdown()

                load_btn.click(load_case, inputs=task_id, outputs=[case_md, truth_md], show_progress="hidden")
                task_id.change(load_case, inputs=task_id, outputs=[case_md, truth_md], show_progress="hidden")
                run_heuristic_btn.click(
                    run_heuristic,
                    inputs=task_id,
                    outputs=[decision_json, breakdown_json, summary_md, penalties_md],
                )
                run_model_btn.click(
                    run_configured,
                    inputs=task_id,
                    outputs=[decision_json, breakdown_json, summary_md, penalties_md],
                )

            with gr.Tab("Benchmark Summary"):
                gr.Markdown(_benchmark_markdown())
                gr.HTML(_render_metrics_table())
                with gr.Row():
                    gr.HTML(_render_svg(REWARD_CHART_PATH))
                    gr.HTML(_render_svg(SCORE_CHART_PATH))
                gr.Markdown(
                    "The trained adapter metrics above come from the successful HF Job `69ecb239d70108f37acde5a1`, "
                    "with committed local copies under `artifacts/phase6/final_results/`."
                )

    return demo


def main() -> None:
    demo = build_demo()
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
