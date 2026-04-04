---
title: Physio Support Openenv
emoji: 💻
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: OpenEnv-style physiotherapy support environment for booking, rescheduling, and escalation
---

# PhysioSupportEnv

PhysioSupportEnv is a small Python environment for a physiotherapy support agent workflow.

It simulates three real-world tasks:

- new home physiotherapy booking
- rescheduling an existing booking
- escalation of urgent or unsafe cases to human support

## Problem Statement

The agent receives a structured support state and must choose the next correct action in the workflow. The environment validates the action, updates the state, and returns a reward.

This project is designed around a simple OpenEnv-style interface:

- `reset()`
- `state()`
- `step(action)`

## Tasks

### 1. New Booking

The patient wants a home physiotherapy session but has not provided a pincode yet.

Expected flow:

- `ask_pincode`
- `show_available_slots`
- `book_slot`
- `confirm_completion`

### 2. Reschedule

The patient already has a booking and wants to move it to another available slot.

Expected flow:

- `show_available_slots`
- `reschedule_slot`
- `confirm_completion`

### 3. Escalation

The request is urgent or unsafe and should not be auto-booked.

Expected flow:

- `escalate_to_human`
- `confirm_completion`

## Observation Format

Each state includes:

- `task_id`
- `task_type`
- `patient_message`
- `known_info`
- `available_slots`
- `conversation_stage`
- `steps_taken`
- `booking_status`
- `flow_progress`
- `allowed_actions`

## Action Space

Supported actions:

- `ask_pincode`
- `ask_time_preference`
- `show_available_slots`
- `book_slot`
- `reschedule_slot`
- `cancel_booking`
- `escalate_to_human`
- `confirm_completion`

## Reward Design

The environment uses shaped rewards:

- partial reward for correct intermediate actions
- higher reward for correct final resolution
- negative reward for invalid or unsafe actions

The final episode score is graded from `0.0` to `1.0`.

## Project Structure

- `app/env.py` - environment logic
- `app/tasks.py` - task definitions
- `app/models.py` - typed models
- `app/grader.py` - deterministic grading
- `inference.py` - LLM runner using OpenAI-compatible client calls
- `server/app.py` - FastAPI server for Space deployment
- `Dockerfile` - container setup
- `requirements.txt` - Python dependencies
- `openenv.yaml` - environment metadata
- `pyproject.toml` - package metadata

## Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variables:

```powershell
$env:HF_TOKEN="your_token"
$env:OPENAI_BASE_URL="https://router.huggingface.co/v1"
$env:OPENAI_MODEL="katanemo/Arch-Router-1.5B:hf-inference"
```

Run:

```bash
python inference.py
```

## Run Server Locally

Install updated dependencies first:

```bash
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Useful endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `GET /state/{session_id}`
- `POST /step/{session_id}`

## Run With Docker

Build:

```bash
docker build -t physio-env .
```

Run:

```powershell
docker run --rm -e HF_TOKEN=$env:HF_TOKEN -e OPENAI_BASE_URL=$env:OPENAI_BASE_URL -e OPENAI_MODEL=$env:OPENAI_MODEL physio-env
```

The container now serves the API on port `7860`. To expose it locally:

```powershell
docker run --rm -p 7860:7860 -e HF_TOKEN=$env:HF_TOKEN -e OPENAI_BASE_URL=$env:OPENAI_BASE_URL -e OPENAI_MODEL=$env:OPENAI_MODEL physio-env
```

## Environment Variables

- `HF_TOKEN` or `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

## Notes

- Do not store tokens in source files.
- The final submission path uses live LLM calls and fails if the API is not configured.
