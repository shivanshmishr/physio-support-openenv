# Submission Notes

Use this folder as the handoff package for judges or reviewers.

## What To Share

- Environment: `openenv.yaml`, `pyproject.toml`, `requirements.txt`, `.env.example`
- Training scripts: `phase55_bootstrap_sft.py`, `phase6_train.py`, `warmup_sft.py`, `train_scaffold.py`, and the core app training modules under `app/`
- Local artifacts: summaries, eval JSONs, reward/loss curves, prompt data, and smoke runs under `artifacts/`
- HF remote logs: raw logs and job metadata under `hf_jobs/`

## Immediate Conclusion Flow

1. Compare `artifacts/phase6/final_results/training_summary.json` with the completed HF job summaries.
2. Use the raw logs in `hf_jobs/` to cite the exact run configuration and training behavior.
3. Pick the final checkpoint based on action accuracy, avg_score, safety, and priority pain recall.
4. Write one narrative summary: what changed, what improved, what regressed, and why the chosen run is the final submission.

## HF Jobs Included

- `phase6_no_upload_v2`: `69ecb239d70108f37acde5a1` (COMPLETED)
  URL: https://huggingface.co/jobs/shivansh9987/69ecb239d70108f37acde5a1
- `phase6_shaped_hybrid`: `69edd68fd2c8bd8662bcfaca` (COMPLETED)
  URL: https://huggingface.co/jobs/shivansh9987/69edd68fd2c8bd8662bcfaca
- `phase6_actionpush_regression`: `69eddd24d70108f37acdffd7` (COMPLETED)
  URL: https://huggingface.co/jobs/shivansh9987/69eddd24d70108f37acdffd7
- `phase6_actionsafe`: `69ede555d2c8bd8662bcfc50` (COMPLETED)
  URL: https://huggingface.co/jobs/shivansh9987/69ede555d2c8bd8662bcfc50

## Run Comparison

Primary local best result, reconstructed from the successful no-upload HF job:

- `phase6_no_upload_v2` represented locally by `artifacts/phase6/final_results`
- `avg_score`: `0.75975`
- `action_accuracy`: `0.70`
- `risk_accuracy`: `1.00`
- `priority_pain_recall`: `1.00`
- `unsafe_action_rate`: `0.00`
- `improvement_avg_score`: `+0.19625`

Best directly comparable 15-case HF run:

- `phase6_shaped_hybrid`
- `avg_score`: `0.71317`
- `action_accuracy`: `0.60`
- `risk_accuracy`: `0.80`
- `priority_pain_recall`: `1.00`
- `unsafe_action_rate`: `0.00`
- `improvement_avg_score`: `+0.16133`

Completed ablations that should not be the final submission:

- `phase6_actionpush_regression`
  `avg_score`: `0.70950`, `improvement_avg_score`: `-0.09300`, `action_accuracy`: `0.60`
- `phase6_actionsafe`
  `avg_score`: `0.70767`, `improvement_avg_score`: `-0.09483`, `action_accuracy`: `0.60`

Conclusion:

- The aggressive action-push reward change did not help.
- The safer action-shaping retry also did not recover the regression.
- The best final result remains the earlier successful run reconstructed under `artifacts/phase6/final_results`.

## Submission Recommendation

Submit the full folder `artifacts/submission_bundle`.

In your summary, present the runs like this:

1. Final selected result: `phase6_no_upload_v2` represented locally by `artifacts/phase6/final_results`.
2. Best comparable HF ablation: `phase6_shaped_hybrid`.
3. Negative ablations: `phase6_actionpush_regression` and `phase6_actionsafe`.

If the judges ask for one final model or run, use:

- `artifacts/phase6/final_results`

If they ask for one reproducible remote training log to point to in the blog, cite:

- `phase6_shaped_hybrid`

Reason:

- `phase6_no_upload_v2` is the strongest quantitative result.
- `phase6_shaped_hybrid` is the strongest directly comparable 15-case HF run.
- `phase6_actionsafe` completed successfully but is not a better final candidate.

## Blog Outline

1. Problem setup: home physio coordination as a structured decision task.
2. Environment design: schema, reward function, safety constraints, task families.
3. Training progression: heuristic baseline -> teacher/bootstrap SFT -> GRPO shaping.
4. Ablations: shaped-hybrid, action-push regression, safer action shaping.
5. Final conclusion: which run you are submitting and why.
