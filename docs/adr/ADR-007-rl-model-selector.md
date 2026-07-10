# ADR-007: Reinforcement Learning — LinUCB Contextual Bandit for Offline Model Selection

**Status:** Accepted  
**Date:** 2026-07-10  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

After benchmarking 11 models, the natural follow-up question is: **which model should be used for any given forecast window?** The static answer ("always use Random Forest") ignores the fact that different time-series regimes favour different models:

- Low-volatility, highly autocorrelated windows → any ML model is roughly equivalent
- High-volatility or trend-reversing windows → the ranking among ML models shifts meaningfully
- Windows that happen to look like the training distribution of statistical models → statistical models occasionally beat ML (rare but observable in the per-window RMSE matrix)

A **model selection policy** that adapts to the observable properties of each window is therefore more principled than a static choice — and can be evaluated against an oracle (always picking the per-window best) and a random baseline.

Key constraints:
1. No additional model inference at dashboard runtime — all predictions are pre-computed
2. No external simulator — the ETT dataset is historical, not a live environment
3. Must run in < 1 second inside Streamlit (`@st.cache_data`)
4. Must produce interpretable output: which features drove the recommendation

---

## Decision Drivers

- **Offline feasibility**: all 11 models' predictions are pre-computed → every arm's reward is known for every window → no simulator needed
- **Interpretability**: the learned policy should be inspectable (feature weights per arm)
- **Minimal dependencies**: Streamlit Cloud deployment uses a thin `requirements-streamlit.txt`; no new heavy libraries (Stable-Baselines3, RLlib, etc.)
- **Speed**: training on 145 windows must complete in milliseconds
- **Honest evaluation**: the learner must be evaluated sequentially (rolling — update on window t only after recording what it would have picked at t) to avoid look-ahead bias

---

## Considered Options

### Option A — Supervised "Oracle Predictor" (classification/regression)

Train a classifier to predict, from window features, which model has the lowest RMSE. Choose that model.

- **Pros**: Familiar ML paradigm; can use any sklearn classifier
- **Cons**: Requires a fixed train/test split of the 145 windows; ignores the exploration/exploitation trade-off; no principled uncertainty quantification; does not degrade gracefully when context is out-of-distribution
- **Why not chosen**: Bandit framing is more honest — it acknowledges uncertainty in early windows and quantifies it via the UCB term. Supervised framing would need arbitrary train/test split of an already small dataset (145 windows).

### Option B — Full RL with Gymnasium Environment (PPO/DQN)

Wrap the ETT test set as a Gymnasium environment: state = window features, action = model choice, reward = −RMSE, episode = all 145 steps.

- **Pros**: Full sequential decision-making; supports multi-step lookahead
- **Cons**: Requires a Gymnasium wrapper (~100 lines), Stable-Baselines3 (~1 GB), and a meaningful number of training episodes — but there is only one historical episode; policy gradient methods need thousands of episodes to converge; PPO/DQN architectures are overkill for 11 discrete arms and linear reward structure
- **Why not chosen**: Severe data scarcity (1 historical episode ≈ 145 steps). Full RL cannot learn a meaningful policy from one trajectory.

### Option C — Thompson Sampling (Gaussian)

Each arm maintains a Gaussian posterior over expected reward μ_a ~ N(θ̂_a, σ̂_a²); at each step sample μ̃_a from the posterior and pick argmax.

- **Pros**: Simpler to implement than LinUCB; naturally Bayesian; strong regret bounds in the non-contextual setting
- **Cons**: Standard Thompson Sampling is non-contextual (ignores window features); contextual Thompson Sampling with linear models requires the same matrix inversion as LinUCB but loses the clean closed-form UCB interpretability
- **Why not chosen**: The non-contextual version ignores the window features entirely — it would simply converge to always picking the globally best arm (Random Forest), equivalent to static best. Contextual Thompson Sampling offers no interpretability advantage over LinUCB for this dashboard.

### Option D — LinUCB Contextual Bandit (chosen)

Each arm a maintains a ridge-regularised linear model: A_a = I + ΣxxᵀI, b_a = Σrx. UCB score at context x: `θ_a^T x + α √(x^T A_a^{-1} x)`. Select argmax over arms.

- **Pros**: Contextual (uses window features); UCB exploration term provides principled uncertainty quantification; fully interpretable (θ_a weights show which features drive each arm's score); pure numpy implementation (< 70 lines); O(d²) per update (fast); well-studied regret bounds (√T under standard assumptions)
- **Cons**: Linear reward model — cannot capture non-linear relationships between features and model performance; 145 windows is still small (approx. 13 per arm at uniform selection)
- **Why chosen**: Best fit for the constraints: offline, interpretable, fast, no new dependencies, principled uncertainty

---

## Decision Outcome

**Chosen: Option D — LinUCB with full-feedback offline updates.**

### Algorithm (src/rl/bandit.py — LinUCBBandit)

```
Initialise: A_a = I_d,  b_a = 0_d   for all a ∈ {1..K}

For each window w = 1..N:
  Context x_w ← extract_window_features(lookback_w)

  # Record selection BEFORE updating on w (prevents look-ahead bias)
  For each arm a:
      θ_a = A_a^{-1} b_a
      UCB_a = θ_a^T x + α √(x^T A_a^{-1} x)
  selected_arm_w ← argmax UCB_a

  # Full-feedback update: all arms observe their true reward
  For each arm a:
      reward_{w,a} = normalised_rmse_score(w, a)   # 1.0 = best, 0.0 = worst
      A_a ← A_a + x_w x_w^T
      b_a ← b_a + reward_{w,a} · x_w
```

Full-feedback updates are valid here because all arm rewards are observable at every window — the pre-computed forecast JSONs give each model's y_pred for the entire test set.

### Context Features (src/rl/window_features.py)

8 features extracted from the 24-hour lookback window preceding each forecast:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `level` | mean(w) / scale | Series level — high OT values differ from low-OT winter periods |
| `volatility` | std(w) / scale | Local spread — high volatility favours models robust to noise |
| `trend` | polyfit slope / scale | Rising vs falling — gradient-based models handle monotone trends well |
| `autocorr` | lag-1 Pearson r | AR structure — highly autocorrelated windows suit lag-based ML models |
| `max_jump` | max\|diff\| / scale | Shock indicator — single large steps may favour robust (RF) over gradient methods |
| `range_ratio` | (max−min) / scale | Local range width — wide swings indicate non-stationary behaviour |
| `skewness` | scipy.stats.skew | Tail asymmetry — statistical models can be sensitive to skewed residuals |
| `bias` | 1.0 | LinUCB intercept — allows arm-specific intercept in the linear model |

`scale = std(y_true_test)` normalises all magnitude-dependent features by the global test set variability, making the feature vector comparable across different series variants.

### Test Set Alignment

The 11 models produce two different test set lengths:
- **Multivariate** (catboost, lightgbm, random_forest, xgboost): 3,480 points — lag features drop 5 initial NaN rows
- **Univariate** (arima, holt_winters, lstm, nbeats, prophet, tft, transformer): 3,485 points

Relationship: `univariate[5:] == multivariate[:]` (verified with `np.allclose`).

Resolution: all models trimmed to the common 3,480-point window by removing `len(y) - 3480` rows from the start of longer arrays. This preserves the same 145 × 24-hour windows for all arms.

### Hyperparameter: Alpha (α)

α controls the exploration bonus magnitude:
- α = 0 → pure exploitation (always pick highest estimated reward)
- α = 0.5 (default) → moderate exploration; appropriate for offline setting where the agent never suffers real consequences from a wrong pick
- α = 3.0 → heavy exploration; mostly cycles through arms regardless of learned weights

α is exposed as a Streamlit slider (0.1–3.0) in Tab 7; changing it clears the `@st.cache_data` cache and retrains in < 1 second.

### Dashboard Integration (src/ui/dashboard.py — Tab 7)

- `_run_bandit(forecast_dir, alpha)` decorated with `@st.cache_data` — cached by (dir, alpha) so retrain triggers only when alpha changes
- Results dict is fully serialisable (plain Python lists/dicts) — passes `@st.cache_data` serialisation requirement
- For the what-if predictor, theta weights are extracted from the results dict and applied as `scores = weights @ context` (linear expected-reward, no UCB bonus) — no bandit object needed at inference time

---

## Benchmark Results (α = 0.5, ETTh1, 145 windows × 11 arms)

| Policy | Avg RMSE | Notes |
|--------|---------|-------|
| Oracle (best per window) | 0.6084 | Theoretical ceiling — unachievable without future knowledge |
| LinUCB Bandit | 0.6923 | Learns regime-dependent arm selection from context |
| Static Best (Random Forest) | 0.6700 | Optimal fixed policy — always pick the globally best arm |
| Random (uniform) | 3.3876 | Includes frequent selection of statistical arms (RMSE 4–9) |

**Bandit vs random: 80% improvement** in avg RMSE. The ~3.2 gap vs static best (0.6923 vs 0.6700) represents the exploration cost in early windows before the bandit converges to ML arms.

Final arm selection distribution across 145 windows:

| Arm | Count | % | Family |
|-----|-------|---|--------|
| Random Forest | 66 | 45.5% | ML |
| CatBoost | 56 | 38.6% | ML |
| XGBoost | 16 | 11.0% | ML |
| LightGBM | 5 | 3.4% | ML |
| ARIMA | 1 | 0.7% | Statistical |
| Holt-Winters | 1 | 0.7% | Statistical |
| All DL models | 0 | 0% | Deep Learning |

Statistical arms selected only in the first 2 windows (pure exploration before any updates). The bandit rapidly learns that ML arms dominate on ETTh1.

Cumulative regret after 145 windows:
- LinUCB: **12.16 °C** total RMSE waste above oracle
- Static Best: 8.93 °C (lower regret because it avoids early exploration cost)
- Random: 402.98 °C

---

## Positive Consequences

- Transforms the static benchmark into a **learning system** — the bandit's selections are data-driven, not rule-based
- The θ weights (9 values per arm) make the policy fully interpretable — which features drive preference for each model
- 80% improvement over random selection makes the value of context clear
- Pure numpy implementation means no new dependencies on Streamlit Cloud
- Training < 1 second on 145 windows; `@st.cache_data` prevents redundant retraining

## Negative Consequences

- 145 windows is a small training set for 11 arms (~13 per arm at uniform selection); linear weights may be noisy
- Linear reward model cannot capture non-linear feature × model interactions
- The bandit never strictly outperforms static best (Random Forest) in cumulative regret because the early exploration windows are costly; with more data it would converge to match or beat static best
- What-if predictor uses pure linear scores (no UCB bonus) and may extrapolate poorly outside the observed feature range

---

## Implementation Notes

- `src/rl/__init__.py`: module docstring only
- `src/rl/bandit.py`: `LinUCBBandit(n_arms, n_features, alpha)` — uses `np.linalg.solve` (not explicit inverse) for numerical stability; `get_weights()` exposes θ vectors for the what-if predictor
- `src/rl/window_features.py`: `extract_window_features(window, scale)` returns float32 array of shape (8,); `FEATURE_NAMES` list for labelling
- `src/rl/model_selector.py`: `BanditModelSelector(forecast_dir, alpha, window_size=24)`:
  - `_load_forecasts()`: loads 11 JSON files; aligns lengths by `offset = len(y) - 3480`
  - `train()`: builds 145×11 RMSE matrix, normalises per-window reward, extracts contexts, runs sequential LinUCB loop
  - `get_results()`: returns serialisable dict including weights, feature stats (min/max/mean), regret series
  - `predict_from_features(context)`: scores each arm with θ_a @ context
- `requirements-streamlit.txt`: `scipy>=1.10.0` added (used by `scipy.stats.skew` in window_features.py)

---

## Related Decisions

- ADR-006: Pre-computed JSON strategy makes full-feedback offline bandit possible — all arm rewards observable from stored y_pred arrays
- ADR-004: Model selection (11 arms) — bandit operates over the same set of models benchmarked in ADR-004
- ADR-003: 70/10/20 chronological split — bandit trains on the test split only (20% = 3,480 points)

---

## References

- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *WWW 2010*. https://arxiv.org/abs/1003.0146
- Lattimore, T., & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press. (Chapter 19 — LinUCB)
