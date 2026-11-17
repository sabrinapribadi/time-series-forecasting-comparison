"""Ask AI (R26): a small retrieval assistant over the benchmark's own results; replay by default, the viewer's own key optional.

Knowledge comes from committed files only: the fair rolling benchmark (ADR-008, data/forecasts/ETT/rolling/summary.json),
the original single-split results (labelled not like-for-like) and the model and metric metadata (meta.py).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import agent as A                                  # shared chat engine: number check, replay matching
import meta

ROOT = Path(__file__).resolve().parents[2]
FORECASTS = ROOT / "data" / "forecasts" / "ETT"
MODEL = "gpt-4o-mini"
PRICES = {"input": 0.15, "cached_input": 0.075, "output": 0.60}   # USD per 1M tokens (developers.openai.com pricing, read 2026-11-11)
FAIR = {"chronos2": "Chronos-2", "chronos2_cov": "Chronos-2 with covariates", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5",
        "holt_winters": "Holt-Winters", "nbeats": "N-BEATS", "random_forest": "Random Forest", "seasonal_naive_24": "Same hour yesterday",
        "arima": "ARIMA", "catboost": "CatBoost", "xgboost": "XGBoost", "lightgbm": "LightGBM", "prophet": "Prophet",
        "seasonal_naive_168": "Same hour last week", "lstm": "LSTM"}
SYSTEM = ("You are a data science assistant inside the ETT forecasting benchmark dashboard. Answer from the context only and cite the "
          "source id in square brackets after each claim that uses it. The fair rolling benchmark (ADR-008) is the result to use for "
          "any ranking or recommendation; the original single-split table is not like-for-like (machine-learning models were given the "
          "true previous values) and must not be used to rank models. Quote numbers as they appear in the context; never compute new "
          "ones such as differences or percentages. Ranks come from the numbered list; a champion is the best of its group "
          "(foundation or trained), not necessarily second overall. If the context does not answer the question, say so and say what the dashboard does cover. At most 150 words.")
QUESTIONS = [
    "Which model should I use for production?",
    "Did the zero-shot foundation model beat the trained models?",
    "Why did the machine-learning models look so good in the original benchmark?",
    "How does LSTM compare with Holt-Winters in the fair benchmark?",
    "What does MASE mean and why is it the main metric?",
    "What is the ETT dataset?",
    "What will the stock market do tomorrow?",
]


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def load_results() -> dict:
    out = {}
    for p in sorted(FORECASTS.glob("*.json")):
        if p.stem.endswith(("_feature_importance", "_shap")):
            continue
        d = json.loads(p.read_text())
        out[p.stem] = {k: v for k, v in d.items() if k not in ("y_true", "y_pred")} | {"_model_key": d.get("model", p.stem.split("_h")[0].split("_m")[0])}
    return out


def knowledge(results: dict | None = None) -> list[dict]:
    """Retrieval chunks: id, tags, text."""
    s = json.loads((FORECASTS / "rolling" / "summary.json").read_text())
    g = s["gate"]
    rows = sorted(((m, v) for m, v in s["models"].items() if "test" in v), key=lambda mv: mv[1]["test"]["mase"])
    fair_lines = [f"  {i}. {FAIR.get(m, m)} ({v['group']}): test MASE {_fmt(v['test']['mase'])}, RMSE {_fmt(v['test']['rmse'])} °C, "
                  f"MAE {_fmt(v['test']['mae'])} °C" for i, (m, v) in enumerate(rows, 1)]
    chunks = [
        {"id": "fair_benchmark", "tags": ["fair", "rolling", "adr-008", "results", "ranking", "best", "production", "champion", "foundation",
                                           "zero-shot", "chronos", "trained", "holt", "winters", "mase", "rmse", "lstm", "compare", "model", "use"],
         "text": (f"Fair rolling benchmark (ADR-008): every model forecasts the next 24 hours of oil temperature from the same history, at "
                  f"{s['origins']['test']} test origins ({s['origins']['val']} validation origins chose the champions first). Ranked by test MASE:\n"
                  + "\n".join(fair_lines)
                  + f"\nChampions chosen on validation: {FAIR[g['foundation_champion']]} (foundation, zero-shot) and {FAIR[g['trained_champion']]} "
                    f"(trained). Test MASE {_fmt(g['test_mase'][g['foundation_champion']])} vs {_fmt(g['test_mase'][g['trained_champion']])}: "
                    f"the foundation model's MASE is {g['foundation_lower_by']:.1%} lower; Diebold–Mariano (HLN) p = {g['dm_hln']['p']:.1e} over "
                    f"{g['dm_hln']['n']} origins. Verdict: {g['verdict']}. Dropped: "
                  + "; ".join(f"{FAIR.get(k, k)} ({v})" for k, v in s["dropped"].items()) + ".")},
        {"id": "fair_vs_original", "tags": ["original", "oracle", "lag", "leak", "leakage", "machine", "learning", "ml", "good", "why", "fair",
                                             "xgboost", "lightgbm", "catboost", "random", "forest"],
         "text": ("Why the original benchmark is not like-for-like: in the original single split, machine-learning models received lag features "
                  "holding the true oil temperature of the previous hours at every test step (look-ahead), while statistical and deep models "
                  "forecast the whole test period in one shot. Original RMSE vs fair rolling RMSE (°C): "
                  + "; ".join(f"{FAIR.get(m, m)} {_fmt(v['old_rmse'], 2)} vs {_fmt(v['rolling_rmse'], 2)}" for m, v in s["old_vs_rolling"].items())
                  + ". The fair benchmark gives every model the same history at each origin.")},
        {"id": "mase", "tags": ["mase", "metric", "scaled", "main", "primary", "why", "mean"],
         "text": (f"MASE (mean absolute scaled error) is the primary metric (ADR-008): average absolute error divided by the in-sample error of "
                  f"'same hour yesterday' on the training data ({_fmt(s['mase_scale'])} °C). Below 1 beats that simple rule. It is comparable "
                  "across origins and was fixed before any test score was read.")},
        {"id": "dataset_overview", "tags": ["dataset", "ett", "overview", "columns", "target", "data", "what"],
         "text": ("ETT (Electricity Transformer Temperature), ETTh1: hourly readings, 17,420 rows, 2016–2018. Target OT is the transformer's "
                  "oil temperature in °C; six load columns (HUFL, HULL, MUFL, MULL, LUFL, LULL). Source: github.com/zhouhaoyi/ETDataset "
                  "(MIT licence); Zhou et al., AAAI 2021 (Informer).")},
    ]
    if results:
        orig = []
        for run_key, d in results.items():
            mk = d.get("_model_key", run_key)
            m = d.get("metrics", {})
            orig.append(f"  {meta.MODEL_DISPLAY.get(mk, mk)} ({meta.FAMILY_MAP.get(mk, 'Unknown')}): RMSE={_fmt(m.get('RMSE'), 4)}, R2={_fmt(m.get('R2'), 4)}")
        chunks.append({"id": "original_benchmark", "tags": ["original", "single", "split", "table", "r2"],
                       "text": "ORIGINAL single-split results (NOT like-for-like; do not use to rank models):\n" + "\n".join(orig)})
    for fam, keys in [("Statistical", ["holt_winters", "arima", "prophet"]), ("ML", ["random_forest", "xgboost", "lightgbm", "catboost"]),
                      ("Deep Learning", ["lstm", "transformer", "nbeats", "tft"])]:
        parts = [f"{meta.MODEL_META.get(k, {}).get('label', k)}: {meta.MODEL_META.get(k, {}).get('description', '')} "
                 f"Strengths: {meta.MODEL_META.get(k, {}).get('strengths', '')}. Weaknesses: {meta.MODEL_META.get(k, {}).get('weaknesses', '')}."
                 for k in keys if k in meta.MODEL_META]
        chunks.append({"id": f"family_{fam.lower().replace(' ', '_')}", "tags": [fam.lower(), "algorithm", "model", "description"] + keys,
                       "text": f"{fam} model family:\n" + "\n".join(parts)})
    for mk, mm in meta.METRIC_META.items():
        chunks.append({"id": f"metric_{mk.lower()}", "tags": [mk.lower(), "metric", "evaluation"],
                       "text": f"{mm['full']} ({mk}): {mm['description']} Formula: {mm['formula']}. Unit: {mm['unit']}."})
    return chunks


def retrieve(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    """Keyword retrieval (tags weigh 3, text hits 1); the fair benchmark is always included."""
    words = {w for w in re.findall(r"[a-z0-9\-]+", query.lower()) if len(w) > 2}
    scored = sorted(chunks, key=lambda c: -(3 * sum(any(w in t for w in words) for t in c["tags"]) + sum(c["text"].lower().count(w) for w in words)))
    top = scored[:top_k]
    if not any(c["id"] == "fair_benchmark" for c in top):
        top[-1] = next(c for c in chunks if c["id"] == "fair_benchmark")
    return top


def live(question: str, key: str | None, results: dict | None = None) -> dict:
    """One answer: retrieve, then gpt-4o-mini with the cited-answer prompt; traced for replay and display."""
    from openai import OpenAI
    t0 = time.perf_counter()
    ctx = retrieve(question, knowledge(results if results is not None else load_results()))
    context = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in ctx)
    r = OpenAI(api_key=key, timeout=30, max_retries=1).responses.create(          # Responses API, as the other portfolio agents
        model=MODEL, instructions=SYSTEM + "\n\nCONTEXT:\n" + context, input=question, max_output_tokens=400, temperature=0)
    text = (getattr(r, "output_text", "") or "").strip()
    u = getattr(r, "usage", None)
    tin, tout = int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)
    cited = set(re.findall(r"\[([a-z0-9_\-]+)\]", text))
    sources = [{"id": c["id"], "text": c["text"], "cited": c["id"] in cited} for c in ctx]
    bad = A.ungrounded_numbers(text, [question, [c["text"] for c in ctx]])
    return {"question": question, "answer": text, "images": [],
            "tool_calls": [{"name": "retrieve_context", "arguments": {"top_k": len(ctx)}, "ok": True}],
            "tool_results": sources, "steps": 1, "model": MODEL, "grounded": not bad and bool(text), "ungrounded_numbers": bad,
            "tokens": {"input": tin, "cached_input": 0, "output": tout},
            "cost_usd": round((tin * PRICES["input"] + tout * PRICES["output"]) / 1e6, 8),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
