"""ADR-008 protocol checks: origins, the clean tail, MASE scale and the Diebold–Mariano test."""

import numpy as np

from src.evaluation import rolling as R


def test_origins_match_the_pre_registration():
    te, va = R.origins("test"), R.origins("val")
    assert len(te) == 145 and te[0] == 13_936 and te[-1] + R.H <= R.N_ROWS
    assert len(va) == 71 and va[0] == 12_218 and va[-1] + R.H <= R.TEST_START      # no validation target in the test split
    assert int(R.clean_mask(te).sum()) == 41


def test_mase_scale_is_seasonal_naive_on_train():
    y = np.tile(np.arange(24.0), 600)                    # perfectly daily series: seasonal naive is exact
    assert R.mase_scale(y) == 0.0
    y = np.arange(20_000.0)
    assert R.mase_scale(y) == 24.0


def test_dm_hln_detects_a_clear_difference_and_not_noise():
    r = np.random.default_rng(0)
    a = r.normal(1.0, 0.1, 145); b = a + 0.2 + r.normal(0, 0.05, 145)
    assert R.dm_hln(a, b)["p"] < 1e-6
    c = a + r.normal(0, 0.05, 145)
    assert R.dm_hln(a, c)["p"] > 0.01


def test_wql_zero_for_perfect_quantiles():
    y = np.ones((3, 24))
    assert R.wql(y, {"0.1": y, "0.5": y, "0.9": y}) == 0.0


def _s(mase, per_origin, clean):
    return {"test": {"mase": mase, "per_origin_mae": per_origin}, "clean_tail": {"mase": clean}}


def test_gate_outcomes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rr", R.ROOT / "scripts" / "rolling_report.py"); rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    r = np.random.default_rng(1); base = r.normal(2.0, 0.2, 145)
    better = base * 0.85 + r.normal(0, 0.01, 145)
    assert rr.gate("fm", "tr", {"fm": _s(0.85, better, 0.8), "tr": _s(1.0, base, 1.0)})["verdict"] == "foundation model wins"
    assert rr.gate("fm", "tr", {"fm": _s(0.85, better, 1.1), "tr": _s(1.0, base, 1.0)})["verdict"] == "no clear winner"   # clean tail disagrees
    assert rr.gate("fm", "tr", {"fm": _s(1.0, base, 1.0), "tr": _s(0.85, better, 0.8)})["verdict"] == "trained model wins"
    close = base + r.normal(0, 0.3, 145)
    assert rr.gate("fm", "tr", {"fm": _s(0.98, close, 0.9), "tr": _s(1.0, base, 1.0)})["verdict"] == "no clear winner"    # under 5%
