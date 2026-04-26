# Phase 4 Baseline Evaluation

- policy: `heuristic`
- split: `eval`
- variants_per_task: `6`
- num_tasks: `10`

## Metrics
- num_cases: `10`
- avg_reward: `0.9605`
- avg_score: `0.9`
- intent_accuracy: `0.6`
- risk_accuracy: `1.0`
- action_accuracy: `0.8`
- callback_correctness: `1.0`
- priority_pain_recall: `1.0`
- unsafe_action_rate: `0.0`
- summary_completeness: `0.64`

## Showcase Examples
- `callback_worsening_pain_eval_v1` (action_miss): score=`0.9`, reward=`0.91`, action_exact=`False`, intent_exact=`False`
- `reschedule_caregiver_access_eval_v1` (intent_miss): score=`0.9`, reward=`0.99`, action_exact=`True`, intent_exact=`False`
- `mixed_intent_priority_reschedule_eval_v1` (priority_case): score=`0.9`, reward=`0.98`, action_exact=`True`, intent_exact=`True`
