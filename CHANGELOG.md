# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [7.0.1] — 2026-10-09 — Neutral wording for tuning-budget comments
**Files:** Makefile, configs/optuna_configs.yaml, scripts/train_model.py
- **Situation:** Comments explaining Optuna trial budgets and LightGBM defaults referred to a non-public source instead of describing the choice itself.
- **Task:** Keep every comment self-explanatory and limited to public references (papers, public datasets).
- **Action:** Reworded 7 comments to state the actual rationale — per-model trial budgets (cheaper models get fewer trials) and fixed, regularised LightGBM defaults. No code, parameter, or result changed.
- **Result:** Zero behaviour change; benchmark numbers unchanged; comments now read as design rationale.
- **Talking point:** "Hyperparameter budgets are a cost decision — I size Optuna trials by model training cost and document that in config, not in tribal knowledge."
