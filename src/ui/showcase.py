"""First screen of the app (R22, R26 item 7): headline results of the fair benchmark (ADR-008) and the brief as compact cards."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/sabrinapribadi/time-series-forecasting-comparison"
PAGES = "https://sabrinapribadi.github.io/time-series-forecasting-comparison/"


def headline() -> dict:
    s = json.loads((ROOT / "data/forecasts/ETT/rolling/summary.json").read_text())
    g = s["gate"]
    f, t = g["foundation_champion"], g["trained_champion"]
    return {"mase_f": g["test_mase"][f], "mase_t": g["test_mase"][t], "lower_by": g["foundation_lower_by"], "p": g["dm_hln"]["p"],
            "origins": s["origins"]["test"], "methods": sum(1 for v in s["models"].values() if "test" in v),
            "mae_f": s["models"][f]["test"]["mae"], "mae_t": s["models"][t]["test"]["mae"]}


def brief() -> list[tuple[str, list[str]]]:
    import frame as F
    return F.load_brief(ROOT, REPO)


def render(st) -> None:
    """Hero, KPI strip, the 5W1H brief and links (shared app frame)."""
    import frame as F
    h = headline()
    F.hero(st, f"ETTh1 transformer oil temperature · 24 hours ahead · {h['origins']} test forecasts",
           f"Forecasting benchmark: {h['methods']} methods, one fair test",
           "Statistics, machine learning, deep learning and zero-shot foundation models, all forecasting from the same history.")
    F.kpis(st, [("Chronos-2 MASE", f"{h['mase_f']:.3f}", "zero-shot"),
                ("Holt-Winters MASE", f"{h['mase_t']:.3f}", f"Chronos-2 −{h['lower_by']:.0%}"),
                ("Chronos-2 miss", f"{h['mae_f']:.2f} °C", f"Holt-Winters {h['mae_t']:.2f} °C"),
                ("Diebold–Mariano", f"p = {h['p']:.0e}", f"{h['origins']} daily forecasts")])
    F.brief(st, brief())
    F.links_line(st, {"Results page": PAGES, "Technical walkthrough": PAGES + "build.html", "Source": REPO})
