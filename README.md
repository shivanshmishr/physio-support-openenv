---
title: Physio Support Openenv
emoji: 🏥
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Home physiotherapy care-coordination environment with structured reward scoring
---

# PhysioSupportEnv

PhysioSupportEnv is an OpenEnv-style RL environment for home physiotherapy care coordination. The agent reads a realistic patient support case, produces one structured care decision, and gets a deterministic reward based on safety, operational correctness, and communication quality.

## What The Environment Trains

- booking correctness
- rescheduling correctness
- callback handling for worsening pain
- priority escalation for severe pain
- concise patient communication with useful therapist summaries

## Current Task Families

- `booking`
- `rescheduling`
- `callback`
- `priority_pain`

The repo currently includes five seeded cases covering standard booking, caregiver and access blockers, worsening-pain callbacks, critical pain escalation, and a mixed-intent pain plus reschedule case.

## OpenEnv Interface

- `reset()`
- `state()`
- `step(action)`

Each episode is one patient-care coordination case. The current Phase 2 setup evaluates one structured decision per episode.

## Observation Schema

Each state includes:

- `task_id`
- `task_family`
- `patient_message`
- `patient_history_summary`
- `care_plan_summary`
- `visit_context`
- `operational_constraints`
- `allowed_actions`
- `policy_constraints`
- `steps_taken`
- `episode_status`

## Output Schema

The model must return JSON in this shape:

```json
{
  "intent": "mixed_intent",
  "risk_level": "high",
  "next_action": "priority_callback",
  "secondary_actions": ["notify_therapist"],
  "patient_reply": "I'm sorry the pain has increased. I'm marking this as a priority callback so our team can contact you quickly.",
  "therapist_summary": "Patient reports worsening pain before the next visit. Priority callback triggered and therapist notified.",
  "risk_flag": "priority_pain_case"
}
```

## Supported Labels

Supported intents:

- `book_visit`
- `reschedule_visit`
- `cancel_visit`
- `request_callback`
- `report_worsening_pain`
- `ask_general_info`
- `caregiver_unavailable`
- `home_access_issue`
- `mixed_intent`

Supported actions:

- `confirm_home_visit`
- `reschedule_home_visit`
- `request_more_information`
- `schedule_callback`
- `priority_callback`
- `notify_therapist`
- `convert_to_remote_checkin`
- `modify_visit_plan`
- `escalate_for_clinical_review`
- `escalate_for_emergency_attention`
- `close_with_guidance`

## Reward Design

The reward engine in [app/grader.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/grader.py) uses weighted components:

- intent correctness: `0.15`
- risk classification correctness: `0.20`
- action correctness: `0.25`
- policy compliance: `0.10`
- logistics validity: `0.10`
- escalation correctness: `0.10`
- summary completeness: `0.05`
- patient reply quality: `0.05`

Penalties are applied for:

- schema failure
- unsafe failure to escalate
- invalid or forbidden actions
- contradiction between reply and selected action
- unnecessary clarification
- wrong risk flag

## Project Structure

- [app/env.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/env.py)
- [app/grader.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/grader.py)
- [app/models.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/models.py)
- [app/case_generator.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/case_generator.py)
- [app/evaluation.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/evaluation.py)
- [app/heuristic_policy.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/heuristic_policy.py)
- [app/model_policy.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/model_policy.py)
- [app/plotting.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/plotting.py)
- [app/prompting.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/prompting.py)
- [app/tasks.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/tasks.py)
- [app/training_data.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/training_data.py)
- [app/structured_output.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/app/structured_output.py)
- [inference.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/inference.py)
- [train_scaffold.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/train_scaffold.py)
- [evaluate.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/evaluate.py)
- [server/app.py](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/server/app.py)
- [openenv.yaml](/c:/Users/91932/OneDrive/Desktop/MetaHackathon/openenv.yaml)

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python inference.py
```

Optional environment variables for live model calls:

```powershell
$env:HF_TOKEN="your_token"
$env:OPENAI_API_KEY="your_token"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="katanemo/Arch-Router-1.5B:hf-inference"
```

Without API credentials, `inference.py` falls back to a deterministic heuristic policy so the environment still runs.

To run local adapter inference after fine-tuning:

```powershell
$env:LOCAL_BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
$env:LOCAL_ADAPTER_PATH="artifacts/training"
python inference.py
```

## Training Scaffold

Run the LoRA fine-tuning scaffold:

```bash
python train_scaffold.py --base-model Qwen/Qwen2.5-0.5B-Instruct --num-train-epochs 3 --variants-per-task 8 --output-dir artifacts/training
```

What it does:

- expands each seeded task into deterministic train and eval variants
- builds an SFT dataset of observation to JSON decision examples
- fine-tunes a causal LM with `TRL` `SFTTrainer` and `PEFT` LoRA
- evaluates the untuned base model and the trained adapter with the same reward engine
- saves adapter checkpoints and reports under `artifacts/training`
- exports `reward_curve.svg` and `loss_curve.svg`

Key outputs:

- `data/train.jsonl`
- `data/eval.jsonl`
- `baseline_eval.json`
- `heuristic_eval.json`
- `trained_eval.json`
- `training_summary.json`
- `trainer_log_history.json`
- `reward_curve.svg`
- `loss_curve.svg`

## Evaluation Script

Evaluate the heuristic baseline:

```bash
python evaluate.py --policy heuristic --split eval --variants-per-task 8
```

Evaluate the untuned base model:

```bash
python evaluate.py --policy baseline --base-model Qwen/Qwen2.5-0.5B-Instruct --split eval --variants-per-task 8
```

Evaluate a trained adapter:

```bash
python evaluate.py --policy trained --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path artifacts/training --split eval --variants-per-task 8
```

## API

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Useful endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `GET /state/{session_id}`
- `POST /step/{session_id}`
- `POST /run_inference/{task_id}`

`POST /step/{session_id}` expects the same structured JSON shown in the output schema section.

## Current Status

Phase 2 and a real LoRA fine-tuning path are now wired:

- structured task observations
- PRD-aligned response schema
- deterministic reward breakdown
- safety penalties for missed escalation
- inference runner and FastAPI endpoints updated to the new contract
- runnable `TRL` + `PEFT` LoRA training scaffold
- shared evaluation metrics for reward, intent, risk, action, callback, priority recall, and unsafe rate
- base-model vs trained-adapter evaluation with the same artifact format
- SVG reward and loss curve export

Next build steps are dependency installation, an actual fine-tuning run to produce committed artifacts, PNG plot export for submission assets, and HF Space packaging.
