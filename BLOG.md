# Physio Support OpenEnv: Training a Safer Home-Physio Coordination Agent

## 1. Problem Statement

Home physiotherapy support is not just a scheduling task. A real support agent has to identify patient intent, detect safety escalation cases, choose the correct operational next step, and communicate clearly to both the patient and the therapist. That means the task combines:

- language understanding
- risk detection
- workflow routing
- constrained action selection
- structured communication

This is exactly the kind of problem where plain next-token fluency is not enough. An answer can sound good and still be operationally wrong.

We built this project as an OpenEnv-compatible environment so a model can be trained on the actual workflow, not just shown static examples.

## 2. Why This Matters In The Real World

We intentionally started with a narrow slice of healthcare operations: home physiotherapy care coordination. That niche was small enough to model during a hackathon, but rich enough to be realistic and safety-sensitive.

The broader significance is much larger.

If a model can learn to reliably handle:

- booking and rescheduling
- callback requests
- pain escalation routing
- structured patient replies
- therapist summaries
- safety-aware action selection

then the same architecture can be extended to a much wider operational layer of healthcare and service industries:

- home healthcare operations
- outpatient coordination desks
- telehealth support workflows
- rehab and therapy scheduling teams
- patient callback routing systems
- support centers where staff need consistent structured decisions

So physiotherapy is not the final scope. It is the first controlled domain in which we prove the training pattern works.

## 3. Environment Design

The environment is implemented in [app/env.py](app/env.py) and exposed through [openenv.yaml](openenv.yaml). Each episode is one realistic patient-support case.

The observation includes structured context such as:

- patient message
- patient history summary
- care plan summary
- visit context
- operational constraints
- allowed actions
- policy constraints

The model must produce structured JSON with:

- `intent`
- `risk_level`
- `next_action`
- `secondary_actions`
- `patient_reply`
- `therapist_summary`
- `risk_flag`

This design matters because real-world operational systems need outputs that are both human-readable and machine-usable.

## 4. Reward Design

The reward function was designed to reflect both clinical safety and operational correctness. We decomposed the score into interpretable parts:

- intent correctness
- risk classification correctness
- action correctness
- policy compliance
- logistics validity
- escalation correctness
- summary completeness
- patient reply quality

We also penalized failure modes that are common in LLM systems:

- invalid schema
- forbidden or unsupported actions
- unsafe missed escalation
- contradictions between chosen action and reply text
- unnecessary clarification
- wrong risk flags

The key principle was simple: the reward should be hard to game. A fluent answer that violates workflow logic should not score well.

## 5. Training Strategy

Because the hackathon execution window was short, we optimized for iteration speed and training evidence quality. Instead of trying to squeeze in a large model with very few successful runs, we chose:

- `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA adapters
- TRL-based training
- multiple small, comparable experiments

This gave us more iteration cycles in the same amount of time and allowed us to improve the environment, reward design, and evaluation process rather than spending the entire budget on model size.

## 6. Training Pipeline

The training pipeline had three stages:

1. Heuristic baseline and evaluation
2. Bootstrap SFT warm start
3. Environment-reward GRPO fine-tuning

The core scripts are:

- [train_scaffold.py](train_scaffold.py)
- [warmup_sft.py](warmup_sft.py)
- [phase55_bootstrap_sft.py](phase55_bootstrap_sft.py)
- [phase6_train.py](phase6_train.py)

### Why this staged approach

- The heuristic teacher gave us a strong structured reference policy.
- Bootstrap SFT taught the model the schema and basic action format.
- GRPO then optimized against the actual environment reward instead of only imitating labels.

This was the most practical path to showing real improvement within limited time.

## 7. What We Actually Did

The practical workflow during the project was:

1. Build seeded cases across booking, rescheduling, callback, and priority-pain workflows.
2. Expand those seeded cases into deterministic train and eval variants.
3. Implement a structured schema for model outputs.
4. Build a deterministic rubric-based reward engine.
5. Evaluate the untuned base model.
6. Evaluate the heuristic teacher.
7. Run bootstrap SFT to stabilize structured output behavior.
8. Run GRPO using real environment rewards.
9. Compare baseline, trained model, and heuristic teacher with the same evaluation path.
10. Run ablations on reward shaping to understand what helped and what regressed.

This produced a complete training story rather than a single isolated run.

## 8. What Improved

Our final selected result is stored under [artifacts/phase6/final_results](artifacts/phase6/final_results).

Held-out comparison:

| Policy | Avg Reward | Avg Score | Risk Acc | Action Acc | Priority Recall | Unsafe Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline Qwen2.5-0.5B | `0.7003` | `0.5635` | `0.70` | `0.60` | `0.50` | `0.00` |
| Final Trained Adapter | `0.7958` | `0.7598` | `1.00` | `0.70` | `1.00` | `0.00` |
| Heuristic Teacher | `0.9605` | `0.9000` | `1.00` | `0.80` | `1.00` | `0.00` |

Main improvements from baseline to trained model:

- avg reward: `+0.0955`
- avg score: `+0.1963`
- risk accuracy: `+0.30`
- action accuracy: `+0.10`
- priority pain recall: `+0.50`
- unsafe action rate stayed at `0.00`

The most important safety result is that the trained adapter improved on the high-risk routing behavior:

- `risk_accuracy` improved from `0.70` to `1.00`
- `priority_pain_recall` improved from `0.50` to `1.00`

That is exactly the kind of improvement we wanted this environment to encourage.

## 9. Ablations And What We Learned

We ran multiple Hugging Face jobs to test reward-shaping variants:

- `phase6_shaped_hybrid`
- `phase6_actionpush_regression`
- `phase6_actionsafe`

The main takeaway was that pushing too hard on exact action shaping hurt the overall policy. The safer retry also did not outperform the earlier successful run.

That gave us a useful conclusion:

- more shaping is not automatically better
- environment-aligned metrics must drive checkpoint selection
- training progress should be judged by behavior, not by one reward term

The best final result remained the earlier successful run reconstructed locally under [artifacts/phase6/final_results](artifacts/phase6/final_results).

The full comparison and logs are packaged in [artifacts/submission_bundle](artifacts/submission_bundle) and summarized in [artifacts/submission_bundle/CURATOR_NOTES.md](artifacts/submission_bundle/CURATOR_NOTES.md).

## 10. What We Achieved In A 24-Hour Window

This is not a finished healthcare operations platform, and it should not be presented that way. What we achieved in the available time is an end-to-end proof that the training setup works:

- OpenEnv-compliant environment
- deterministic structured reward
- runnable training scripts
- real training artifacts
- public Space deployment
- multiple experiments, including failed ablations
- measurable improvement on held-out cases

That is a strong hackathon outcome because it demonstrates both engineering execution and a credible training signal.

## 11. Future Scope

There is significant room to improve this environment beyond the hackathon version.

### Environment expansion

- add more task families such as billing confusion, insurance issues, cancellations, therapist availability changes, and equipment logistics
- convert the current one-step setup into longer multi-step episodes
- introduce stronger partial observability over time

### Better reward modeling

- improve patient-reply quality scoring
- improve summary completeness scoring
- define richer acceptable-action sets
- harden the reward against subtle exploit patterns

### Better training

- larger case banks
- more GRPO sweeps
- better bootstrap teachers
- multi-turn training rather than only single structured decisions

### Broader real-world deployment scope

- home healthcare operations
- therapy centers
- telehealth support teams
- patient navigation systems
- callback prioritization queues
- structured clinician-support workflows

The long-term value is not only that a model can answer patients. It is that a model can support human operations teams with safer, more consistent, and more structured decisions.

## 12. Deliverables

- Hugging Face Space: `https://huggingface.co/spaces/shivansh9987/physio-support-openenv`
- Environment manifest: [openenv.yaml](openenv.yaml)
- Training notebook: [phase6_training_notebook.ipynb](phase6_training_notebook.ipynb)
- Training scripts: [phase6_train.py](phase6_train.py), [phase55_bootstrap_sft.py](phase55_bootstrap_sft.py), [train_scaffold.py](train_scaffold.py), [warmup_sft.py](warmup_sft.py)
- Final result bundle: [artifacts/phase6/final_results](artifacts/phase6/final_results)
- Submission package: [artifacts/submission_bundle](artifacts/submission_bundle)

## 13. Final Takeaway

We started with a niche, but the actual contribution is broader: a structured, safety-aware environment for training LLMs on operational healthcare coordination tasks.

Within a short hackathon timeline, we showed that a small model can improve meaningfully when trained against a carefully designed environment and reward function. That makes this project both a concrete submission and a foundation for a larger class of industry-relevant workflow agents.
