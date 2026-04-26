from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "artifacts" / "submission_bundle"

FILES_TO_COPY = [
    "README.md",
    "BLOG.md",
    "phase6_training_notebook.ipynb",
    "openenv.yaml",
    "pyproject.toml",
    "requirements.txt",
    ".env.example",
    "phase55_bootstrap_sft.py",
    "phase6_train.py",
    "warmup_sft.py",
    "train_scaffold.py",
    "evaluate.py",
    "inference.py",
    "app/env.py",
    "app/prompting.py",
    "app/rl_training.py",
    "app/structured_output.py",
    "app/training_data.py",
]

ARTIFACTS_TO_COPY = [
    "artifacts/phase6/final_results",
    "artifacts/phase6/grpo_smoke",
    "artifacts/phase6/training_smoke",
    "artifacts/phase6/actionsafe_hf_job.md",
    "artifacts/phase4/heuristic_baseline",
]

KNOWN_HF_JOBS = [
    {
        "name": "phase6_no_upload_v2",
        "job_id": "69ecb239d70108f37acde5a1",
        "note": "Successful HF Phase 6 run reconstructed into local final_results.",
    },
    {
        "name": "phase6_shaped_hybrid",
        "job_id": "69edd68fd2c8bd8662bcfaca",
        "note": "Earlier shaped-hybrid run discussed during tuning.",
    },
    {
        "name": "phase6_actionpush_regression",
        "job_id": "69eddd24d70108f37acdffd7",
        "note": "Aggressive action-push run that regressed on action accuracy and avg_score.",
    },
    {
        "name": "phase6_actionsafe",
        "job_id": "69ede555d2c8bd8662bcfc50",
        "note": "Current safer action-shaping run submitted from pushed main.",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_path(rel_path: str, destination_root: Path) -> dict[str, Any] | None:
    source = ROOT / rel_path
    if not source.exists():
        return None

    destination = destination_root / rel_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return {
            "path": rel_path,
            "type": "directory",
        }

    shutil.copy2(source, destination)
    return {
        "path": rel_path,
        "type": "file",
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def build_env_manifest() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "safe_env_files": [
            path
            for path in ["openenv.yaml", "pyproject.toml", "requirements.txt", ".env.example"]
            if (ROOT / path).exists()
        ],
        "secret_files_excluded": [
            ".env",
        ],
    }


def fetch_hf_job_bundle(api: HfApi, job_spec: dict[str, str], output_root: Path) -> dict[str, Any]:
    job_id = job_spec["job_id"]
    output_dir = output_root / f"{job_spec['name']}_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = api.inspect_job(job_id=job_id)
    logs = "".join(api.fetch_job_logs(job_id=job_id, follow=False))

    command = info.command or []
    command_text = ""
    if isinstance(command, list):
        command_text = "\n".join(command)

    (output_dir / "logs.txt").write_text(logs, encoding="utf-8")
    (output_dir / "command.txt").write_text(command_text, encoding="utf-8")

    info_payload = {
        "id": info.id,
        "url": info.url,
        "created_at": info.created_at,
        "docker_image": info.docker_image,
        "command": info.command,
        "arguments": info.arguments,
        "environment": info.environment,
        "secrets": info.secrets,
        "flavor": info.flavor,
        "labels": info.labels,
        "status": info.status,
        "owner": info.owner,
        "endpoint": info.endpoint,
        "note": job_spec["note"],
    }
    write_json(output_dir / "job_info.json", info_payload)

    return {
        "name": job_spec["name"],
        "job_id": job_id,
        "status": getattr(info.status, "stage", None),
        "url": info.url,
        "bundle_dir": str(output_dir.relative_to(BUNDLE_DIR)),
    }


def build_curator_notes(job_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Submission Notes",
        "",
        "Use this folder as the handoff package for judges or reviewers.",
        "",
        "## What To Share",
        "",
        "- Environment: `openenv.yaml`, `pyproject.toml`, `requirements.txt`, `.env.example`",
        "- Training scripts: `phase55_bootstrap_sft.py`, `phase6_train.py`, `warmup_sft.py`, `train_scaffold.py`, and the core app training modules under `app/`",
        "- Local artifacts: summaries, eval JSONs, reward/loss curves, prompt data, and smoke runs under `artifacts/`",
        "- HF remote logs: raw logs and job metadata under `hf_jobs/`",
        "",
        "## Immediate Conclusion Flow",
        "",
        "1. Compare `artifacts/phase6/final_results/training_summary.json` with the completed HF job summaries.",
        "2. Use the raw logs in `hf_jobs/` to cite the exact run configuration and training behavior.",
        "3. Pick the final checkpoint based on action accuracy, avg_score, safety, and priority pain recall.",
        "4. Write one narrative summary: what changed, what improved, what regressed, and why the chosen run is the final submission.",
        "",
        "## HF Jobs Included",
        "",
    ]
    for job in job_summaries:
        lines.append(f"- `{job['name']}`: `{job['job_id']}` ({job['status']})")
        lines.append(f"  URL: {job['url']}")
    lines.extend(
        [
            "",
            "## Blog Outline",
            "",
            "1. Problem setup: home physio coordination as a structured decision task.",
            "2. Environment design: schema, reward function, safety constraints, task families.",
            "3. Training progression: heuristic baseline -> teacher/bootstrap SFT -> GRPO shaping.",
            "4. Ablations: shaped-hybrid, action-push regression, safer action shaping.",
            "5. Final conclusion: which run you are submitting and why.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    copied_items: list[dict[str, Any]] = []
    for rel_path in FILES_TO_COPY + ARTIFACTS_TO_COPY:
        copied = copy_path(rel_path, BUNDLE_DIR)
        if copied is not None:
            copied_items.append(copied)

    env_manifest = build_env_manifest()
    write_json(BUNDLE_DIR / "env_manifest.json", env_manifest)

    api = HfApi()
    hf_job_summaries = [fetch_hf_job_bundle(api, job_spec, BUNDLE_DIR / "hf_jobs") for job_spec in KNOWN_HF_JOBS]
    write_json(BUNDLE_DIR / "hf_jobs" / "index.json", hf_job_summaries)

    notes = build_curator_notes(hf_job_summaries)
    (BUNDLE_DIR / "CURATOR_NOTES.md").write_text(notes, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(BUNDLE_DIR),
        "copied_items": copied_items,
        "hf_jobs": hf_job_summaries,
    }
    write_json(BUNDLE_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
