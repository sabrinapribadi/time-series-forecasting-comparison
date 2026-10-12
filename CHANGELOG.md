# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

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
