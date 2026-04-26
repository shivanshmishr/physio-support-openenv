from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_line(x_values, y_values, title: str, x_label: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    plt.plot(x_values, y_values, marker="o", linewidth=2.4, color="#0f766e")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_bars(labels, values, title: str, y_label: str, output_path: Path, color: str) -> None:
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(y_label)
    plt.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def generate_grpo_smoke_curves() -> None:
    history_path = ROOT / "artifacts" / "phase6" / "grpo_smoke" / "trainer_log_history.json"
    history = load_json(history_path)

    reward_points = [entry for entry in history if "reward" in entry and "step" in entry]
    loss_points = [entry for entry in history if "loss" in entry and "step" in entry]

    if reward_points:
        plot_line(
            [entry["step"] for entry in reward_points],
            [entry["reward"] for entry in reward_points],
            "GRPO Reward Curve",
            "Training Step",
            "Reward",
            ROOT / "artifacts" / "phase6" / "grpo_smoke" / "reward_curve.png",
        )

    if loss_points:
        plot_line(
            [entry["step"] for entry in loss_points],
            [entry["loss"] for entry in loss_points],
            "GRPO Loss Curve",
            "Training Step",
            "Loss",
            ROOT / "artifacts" / "phase6" / "grpo_smoke" / "loss_curve.png",
        )


def generate_final_result_bars() -> None:
    summary_path = ROOT / "artifacts" / "phase6" / "final_results" / "training_summary.json"
    summary = load_json(summary_path)

    labels = ["Baseline", "Trained", "Heuristic"]
    reward_values = [
        summary["baseline_metrics"]["avg_reward"],
        summary["trained_metrics"]["avg_reward"],
        summary["heuristic_metrics"]["avg_reward"],
    ]
    score_values = [
        summary["baseline_metrics"]["avg_score"],
        summary["trained_metrics"]["avg_score"],
        summary["heuristic_metrics"]["avg_score"],
    ]

    plot_bars(
        labels,
        reward_values,
        "Held-Out Reward Comparison",
        "Average Reward",
        ROOT / "artifacts" / "phase6" / "final_results" / "reward_comparison.png",
        "#0369a1",
    )
    plot_bars(
        labels,
        score_values,
        "Held-Out Score Comparison",
        "Average Score",
        ROOT / "artifacts" / "phase6" / "final_results" / "score_comparison.png",
        "#7c3aed",
    )


def main() -> None:
    generate_grpo_smoke_curves()
    generate_final_result_bars()


if __name__ == "__main__":
    main()
