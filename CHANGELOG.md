# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [7.1.0] — 2026-11-11 — CI on every push
**Files:** tests/test_forecasts.py, .github/workflows/ci.yml; empty tests/test_models.py and tests/test_data_preprocessing.py removed
- **Situation:** No CI; the two test files were empty.
- **Task:** CI on every push (rule R19) that checks the committed forecasts the dashboard shows.
- **Action:** Contract tests recompute RMSE from each model's stored predictions, check all 11 models exist, and check the README's RMSE table and ranking; error-level lint.
- **Result:** 13 tests pass; stored and recomputed RMSE agree for all 11 models; random forest leads (0.7035).
- **Talking point:** The benchmark table can no longer drift from the predictions behind it.

## [7.0.2] — 2026-10-12 — MIT license
**Files:** LICENSE
- **Situation:** The repository had an empty LICENSE file, so by default nobody else could legally reuse the code.
- **Task:** Make the portfolio code reusable with credit, consistently across all portfolio repositories.
- **Action:** Added the standard MIT license (© 2026 Sabrina Pribadi). Data sources keep their own terms; the license covers code only.
- **Result:** GitHub now detects the license as MIT; no code changed.
- **Talking point:** "All my portfolio code is MIT-licensed, so anyone can reuse it with attribution."

## [7.0.1] — 2026-10-09 — Neutral wording for tuning-budget comments
**Files:** Makefile, configs/optuna_configs.yaml, scripts/train_model.py
- **Situation:** Comments explaining Optuna trial budgets and LightGBM defaults referred to a non-public source instead of describing the choice itself.
- **Task:** Keep every comment self-explanatory and limited to public references (papers, public datasets).
- **Action:** Reworded 7 comments to state the actual rationale — per-model trial budgets (cheaper models get fewer trials) and fixed, regularised LightGBM defaults. No code, parameter, or result changed.
- **Result:** Zero behaviour change; benchmark numbers unchanged; comments now read as design rationale.
- **Talking point:** "Hyperparameter budgets are a cost decision — I size Optuna trials by model training cost and document that in config, not in tribal knowledge."
