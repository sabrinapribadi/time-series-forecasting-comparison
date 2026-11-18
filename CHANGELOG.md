# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [8.4.0] — 2026-11-18 — App template v2: sidebar pages, a business question per page
**Files:** src/ui/dashboard.py, src/ui/showcase.py, src/ui/frame.py (new, shared), tests/test_streamlit_app.py, README.md
- **Situation:** The user wanted every Streamlit app to combine the older dashboards' structure (sidebar pages: a real use case, EDA, an AI assistant; a business question on every view) with the newer apps' 5W1H brief and KPI first screen.
- **Task:** Rebuild this app on one shared template (R26 item 8) and check every page against the data before release.
- **Action:** The eight tabs became nine sidebar pages on the shared `frame.py` (Overview first); each page's answer computed from committed results; the viewer's key kept across pages for the inspector's Explain button.
- **Result:** 7 app tests pass. Checking the pages replaced a false correlation caption (it claimed load and oil temperature correlate above 0.7; the data gives at most 0.22) with computed values, made the actual-temperature line visible on dark, and corrected a docstring that named the wrong model provider.
- **Talking point:** "Every portfolio app now reads the same way: what it is for, the data, the use case, then an assistant; and every page answers a business question from the data."

## [8.3.0] — 2026-11-18 — Dashboard without the owner's key (R26): replay by default, the viewer's own key optional
**Files:** src/ui/dashboard.py, src/ui/meta.py (new, metadata moved out of the dashboard), src/ui/ask_ai.py (new), src/ui/showcase.py (new), src/ui/agent.py and src/ui/chat.py (new, shared with the other portfolio apps), src/ui/replays_ask_ai.json (new), scripts/record_app_replays.py (new), tests/test_streamlit_app.py (new), requirements.txt, requirements-streamlit.txt, .github/workflows/ci.yml, README.md
- **Situation:** Ask AI and "Explain performance" read the owner's OpenAI key from Streamlit secrets, against R26; Ask AI's knowledge described only the original, not like-for-like benchmark (it said ML models dominate, RMSE 0.74–0.85).
- **Task:** Remove the owner's key, show the assistant at $0, and make its answers agree with the fair benchmark.
- **Action:** Knowledge base rebuilt from committed files with the fair rolling benchmark first and the original table labelled not like-for-like; retrieval always includes the fair result; 7 recorded answers replayed with sources, number check, cost and time; a bring-your-own-key live mode (also used by "Explain performance"); a showcase first screen; AppTest in CI on the deployed requirements.
- **Result:** 7 of 7 recorded answers pass the number check; about $0.28 per 1,000 questions. Reading the first recording found two errors that were fixed in the instructions before release: a computed percentage (68.3%, caught by the number check) and Holt-Winters called "second-best" (it is the best trained model, fifth overall).
- **Talking point:** "The assistant used to answer from the leaky benchmark; it now cites the fair one, and the app no longer spends our key."

## [8.2.0] — 2026-11-11 — Dashboard leads with the fair benchmark
**Files:** src/ui/dashboard.py, docs/brief.yaml, README.md, docs/index.html, docs/build.html
- **Situation:** The live dashboard still opened on the original, not like-for-like benchmark.
- **Task:** Make the first tab the fair result without losing the original.
- **Action:** New first tab: headline numbers, ranking with hover details, and a day slider of Chronos-2 vs Holt-Winters vs actual; the old tab renamed and flagged.
- **Result:** The app's first screen matches the README and pages; checked with Streamlit's test harness (no exceptions).
- **Talking point:** "Every surface now leads with the same fair result."

## [8.1.0] — 2026-11-11 — Results page and technical walkthrough (R3, R14), browser tests and gated Pages (R19)
**Files:** docs/index.html (new), docs/build.html (new), docs/assets/preview*.png (new), tests/browser/ (new), .github/workflows/ci.yml, docs/brief.yaml, README.md
- **Situation:** The fair benchmark lived only in the README and ADR-008; the dashboard still shows the old one.
- **Task:** A results page readable in two minutes and a phase-by-phase walkthrough, tested in a real browser before publishing.
- **Action:** Results page with the brief, problem, data, approach, a day-by-day what-if (Chronos-2 vs Holt-Winters on all 145 test days) and the ranking; walkthrough of six phases including two failures; share previews; hover and walkthrough tests in CI; Pages deploy only after tests pass.
- **Result:** 186 results-page and 210 walkthrough checks pass; the page is 4.7 screens on desktop.
- **Talking point:** "Drag through any of 145 test days and see how far a free pretrained model and the best trained model were from what happened."

## [8.0.0] — 2026-11-11 — Fair rolling benchmark: zero-shot foundation models beat every trained model (ADR-008)
**Files:** docs/adr/ADR-008 (new), src/evaluation/rolling.py, scripts/rolling_trained.py, scripts/rolling_foundation.py, scripts/rolling_report.py, scripts/rolling_latency.py, tests/test_rolling.py, data/forecasts/ETT/rolling/, docs/brief.yaml, scripts/brief.py, README.md, docs/PRD.md
- **Situation:** The headline benchmark gave ML the true past values at every step while other models forecast 3,484 hours in one go, so it compared information, not skill. Pretrained forecasting models had arrived.
- **Task:** One fair test for every model, zero-shot foundation models included, with the pass mark, champions and leakage handling fixed before any run.
- **Action:** 145 daily 24-hour forecasts from the same history; Chronos-2, TimesFM-2.5 and Moirai-2.0 on CPU; champions by validation; Diebold–Mariano test; a clean tail no documented training set reaches. Found broken checkpoints (untrained deep models, mislabelled ML) and retrained them; Transformer and TFT dropped after 2 hours, logged before any test score.
- **Result:** Chronos-2 errs 19% less than the best trained model, Holt-Winters (1.13 vs 1.40 °C; p = 1e-07; also on the clean tail). Without true past values, ML errors grow 3.3–3.5×.
- **Talking point:** "Once every model saw the same history, a free pretrained model beat everything we had trained, and the old ML lead turned out to be the protocol."

## [7.2.0] — 2026-11-11 — Evidence tables for every cited paper (R2)
**Files:** docs/references.yaml (new), scripts/evidence.py (new), docs/adr/ADR-*.md, data/outputs/references_evidence.json, CHANGELOG.md
- **Situation:** ADRs listed papers without venue, peer review or citation counts.
- **Task:** Give every cited paper the R2 evidence row, generated by script, never typed.
- **Action:** One references file; OpenAlex fetch for venue, review status, year and citations; tables written into each ADR. ADR-004 now cites the paper behind each of the 11 models it compares, not only the tuning library.
- **Result:** Every paper cited in an ADR shows its evidence, with the retrieval date.
- **Talking point:** "Each method choice shows where its paper was published and how widely it is cited."

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
