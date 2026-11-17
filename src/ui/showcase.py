"""First screen of the app (R22, R26 item 7): headline results of the fair benchmark (ADR-008) and the brief as compact cards."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import yaml

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
    b = yaml.safe_load((ROOT / "docs/brief.yaml").read_text())
    return [(i["w"], [str(x) for x in i["lines"]]) for i in b["brief"]]


def md_html(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank">\1</a>', t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)


CSS = """<style>
.hero-eyebrow{font-size:.76rem;letter-spacing:.09em;text-transform:uppercase;font-weight:600;opacity:.65;margin:0}
.hero-title{font-size:clamp(2rem,4.2vw,3rem);font-weight:800;line-height:1.05;margin:.2rem 0 .4rem;color:#E6EDF3}
.hero-sub{font-size:1.02rem;opacity:.8;max-width:72ch;margin:0 0 .4rem}
.brief-w{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:#79C0FF;margin-bottom:.2rem}
.brief-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin:.2rem 0 .3rem}
.bcard{border:1px solid #30363D;border-radius:10px;padding:8px 12px;font-size:.86rem;line-height:1.4;background:#161B22;color:#C9D1D9}
.bcard a{color:#79C0FF}
</style>"""


def render(st) -> None:
    h = headline()
    st.html(CSS + f"""<p class="hero-eyebrow">ETTh1 transformer oil temperature · 24 hours ahead · {h['origins']} test forecasts</p>
<h1 class="hero-title">Forecasting benchmark: {h['methods']} methods, one fair test</h1>
<p class="hero-sub">Statistics, machine learning, deep learning and zero-shot foundation models, all forecasting from the same history.</p>""")
    k = st.columns(4)
    k[0].metric("Chronos-2 MASE", f"{h['mase_f']:.3f}", "zero-shot", delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Holt-Winters MASE", f"{h['mase_t']:.3f}", f"Chronos-2 −{h['lower_by']:.0%}", delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Chronos-2 miss", f"{h['mae_f']:.2f} °C", f"Holt-Winters {h['mae_t']:.2f} °C", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Diebold–Mariano", f"p = {h['p']:.0e}", f"{h['origins']} daily forecasts", delta_color="off", delta_arrow="off", border=True)
    items = brief()
    st.html('<div class="brief-grid">' + "".join(f'<div class="bcard"><div class="brief-w">{w}</div><div>{md_html(lines[0])}</div></div>'
                                               for w, lines in items) + "</div>")
    with st.expander("Full brief"):
        for w, lines in items:
            st.markdown(f"**{w}:** " + " ".join(lines))
    st.caption(f"[Results page]({PAGES}) · [Technical walkthrough]({PAGES}build.html) · [Source]({REPO}) · Free Streamlit app: it sleeps "
               f"after 12 hours without visits; the [results page]({PAGES}) is always on and carries every result.")
