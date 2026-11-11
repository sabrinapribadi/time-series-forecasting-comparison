"""Contract tests for the committed forecasts the dashboard reads (rule R19: CI on every push).

Stored metrics must equal metrics recomputed from the stored predictions, and the README's ranking must hold.
"""
import json
from pathlib import Path

import numpy as np
import pytest

FORECASTS = sorted((Path(__file__).resolve().parents[1] / "data" / "forecasts" / "ETT").glob("*_h1.json"))
README_RMSE = {"random_forest": 0.703, "catboost": 0.744, "lightgbm": 0.755, "xgboost": 0.756, "transformer": 3.525,
               "prophet": 4.030, "tft": 4.214, "nbeats": 4.358, "lstm": 5.332, "holt_winters": 7.851, "arima": 8.774}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_all_eleven_models_present():
    assert {load(p)["model"] for p in FORECASTS} == set(README_RMSE)


@pytest.mark.parametrize("path", FORECASTS, ids=lambda p: p.stem)
def test_stored_rmse_matches_predictions_and_readme(path):
    j = load(path)
    yt, yp = np.asarray(j["y_true"]), np.asarray(j["y_pred"])
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    assert len(yt) == len(yp) > 0 and abs(rmse - j["metrics"]["RMSE"]) < 1e-6
    assert abs(j["metrics"]["RMSE"] - README_RMSE[j["model"]]) <= 0.001        # README rounds to 3 decimals


def test_tree_models_lead_as_reported():
    best = min((load(p) for p in FORECASTS), key=lambda j: j["metrics"]["RMSE"])
    assert best["model"] == "random_forest"
