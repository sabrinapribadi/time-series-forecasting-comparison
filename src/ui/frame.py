"""Shared app frame for every portfolio Streamlit app (R26 items 1, 4 and 7; app template v2).

One layout for all apps, combining the older dashboards (sidebar navigation, a stats panel, a business question on every page)
with the newer showcase (title, KPI strip, the 5W1H brief on the first screen, free-tier note, always-on fallback link):

    sidebar   brand · page buttons · at-a-glance stats · page controls · links, data licence, free-tier note
    Overview  hero · 3-4 KPI tiles · brief cards (What, Why, Who, Where, When it fails, How) · "use it" cards to every page
    any page  title · what this page shows · the business question it answers, with the answer from the data

Identical in every repository (tests compare a checksum); each app passes its own accent colour and content.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

FREE_TIER = ("Free Streamlit Community Cloud app: it sleeps after 12 hours without visits and the first visit wakes it "
             "(up to a minute); the results page is always on and carries every result.")


@dataclass(frozen=True)
class Page:
    key: str        # URL value: ?page=<key>
    label: str      # sidebar button and page title
    icon: str       # Material icon, e.g. ":material/map:"
    what: str       # one line: what the viewer can do here (overview card and page context)


def md_html(text: str) -> str:
    """Brief markdown (links, bold) to safe inline HTML."""
    t = html.escape(str(text), quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank">\1</a>', t)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)


def load_brief(root: Path, repo: str) -> list[tuple[str, list[str]]]:
    """docs/brief.yaml (R22) as (heading, lines); repository-relative links made absolute."""
    import yaml
    b = yaml.safe_load((Path(root) / "docs" / "brief.yaml").read_text(encoding="utf-8"))
    fix = lambda t: re.sub(r"\]\((?!https?://)([^)]+)\)", lambda m: f"]({repo}/blob/main/{m.group(1)})", t)  # noqa: E731
    return [(i["w"], [fix(str(x)) for x in i["lines"]]) for i in b["brief"]]


def setup(st, accent: str, title_font: str = "'Source Sans 3',system-ui,sans-serif") -> None:
    """CSS for the frame; colours derive from one accent so every app keeps its own identity."""
    st.html(f"""<style>
:root{{--acc:{accent}}}
.f-eyebrow{{font-size:.74rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600;opacity:.65;margin:0}}
.f-title{{font-family:{title_font};font-size:clamp(2rem,4.2vw,3.1rem);font-weight:700;line-height:1.04;margin:.15rem 0 .4rem;letter-spacing:-.01em}}
.f-sub{{font-size:1.02rem;opacity:.8;max-width:72ch;margin:0 0 .35rem}}
.f-ptitle{{font-family:{title_font};font-size:clamp(1.6rem,3vw,2.2rem);font-weight:700;line-height:1.1;margin:.1rem 0 .5rem}}
.f-w{{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700;color:var(--acc);margin-bottom:.2rem}}
.f-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px;margin:.2rem 0 .35rem}}
.f-card{{border:1px solid color-mix(in srgb,var(--acc) 22%,#2c2c2c);border-radius:10px;padding:8px 12px;font-size:.86rem;line-height:1.42;
  background:color-mix(in srgb,var(--acc) 5%,transparent)}}
.f-card a{{color:var(--acc)}}
.f-context{{border-left:3px solid var(--acc);background:color-mix(in srgb,var(--acc) 9%,transparent);padding:.55rem .9rem;
  border-radius:0 8px 8px 0;font-size:.92rem;line-height:1.5;margin:.1rem 0 .5rem}}
.f-biz{{border:1px solid color-mix(in srgb,var(--acc) 30%,#2c2c2c);border-radius:10px;padding:.6rem .9rem;margin:.2rem 0 .8rem}}
.f-biz .q{{font-weight:650;margin:.1rem 0 .25rem}}
.f-biz .a{{opacity:.9;line-height:1.5}}
.f-brand{{display:flex;align-items:center;gap:.6rem;margin:.1rem 0 .2rem}}
.f-brand .n{{font-weight:700;font-size:1.05rem;line-height:1.1}}
.f-brand .t{{font-size:.76rem;opacity:.65;line-height:1.25}}
.f-label{{font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700;opacity:.55;margin:.6rem 0 .3rem}}
.f-stats{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.f-stat{{border:1px solid color-mix(in srgb,var(--acc) 18%,#2c2c2c);border-radius:8px;padding:6px 8px}}
.f-stat .v{{font-weight:700;font-size:1rem;color:var(--acc);font-variant-numeric:tabular-nums}}
.f-stat .l{{font-size:.7rem;opacity:.7;line-height:1.25}}
.f-note{{border-left:3px solid var(--acc);padding:.6rem .9rem;border-radius:0 .5rem .5rem 0;background:color-mix(in srgb,var(--acc) 10%,transparent);line-height:1.5}}
</style>""")


def keep(st, keys: list[str]) -> None:
    """Keep widget values across pages: Streamlit drops a widget's state on runs that do not draw it, unless it is re-set."""
    for k in keys:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _go(key: str) -> None:
    import streamlit as st
    st.session_state["page"] = key


def navigate(st, pages: list[Page], brand: str, tagline: str, icon: str) -> Page:
    """Sidebar brand and page buttons; the open page lives in session state and the URL (?page=), so pages can be linked."""
    keys = [p.key for p in pages]
    if st.session_state.get("page") not in keys:
        q = st.query_params.get("page")
        st.session_state["page"] = q if q in keys else keys[0]
    with st.sidebar:
        st.html(f'<div class="f-brand"><span style="font-size:1.5rem;color:var(--acc)">{icon}</span>'
                f'<div><div class="n">{html.escape(brand)}</div><div class="t">{html.escape(tagline)}</div></div></div>')
        for p in pages:
            st.button(p.label, icon=p.icon, key=f"nav_{p.key}", width="stretch", on_click=_go, args=(p.key,),
                      type="primary" if st.session_state["page"] == p.key else "secondary")
    st.query_params["page"] = st.session_state["page"]
    return next(p for p in pages if p.key == st.session_state["page"])


def sidebar_stats(st, stats: list[tuple[str, str]], label: str = "At a glance") -> None:
    """A 2-column grid of committed results (value, label)."""
    with st.sidebar:
        st.html(f'<div class="f-label">{label}</div><div class="f-stats">' + "".join(
            f'<div class="f-stat"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(lbl)}</div></div>' for v, lbl in stats) + "</div>")


def sidebar_label(st, text: str) -> None:
    with st.sidebar:
        st.html(f'<div class="f-label">{html.escape(text)}</div>')


def sidebar_footer(st, links: dict[str, str], data_note: str) -> None:
    with st.sidebar:
        st.divider()
        st.markdown(" · ".join(f"[{k}]({v})" for k, v in links.items()))
        st.caption(data_note)
        st.caption(FREE_TIER)


def hero(st, eyebrow: str, title: str, sub: str) -> None:
    st.html(f'<p class="f-eyebrow">{eyebrow}</p><h1 class="f-title">{title}</h1><p class="f-sub">{sub}</p>')


def kpis(st, items: list[tuple[str, str, str]]) -> None:
    """3-4 KPI tiles: (label, value, note)."""
    cols = st.columns(len(items))
    for c, (label, value, note) in zip(cols, items):
        c.metric(label, value, note, delta_color="off", delta_arrow="off", border=True)


def brief(st, items: list[tuple[str, list[str]]]) -> None:
    """The 5W1H brief as compact cards (first line each), the full brief one click away."""
    st.html('<div class="f-grid">' + "".join(f'<div class="f-card"><div class="f-w">{w}</div><div>{md_html(lines[0])}</div></div>'
                                          for w, lines in items) + "</div>")
    with st.expander("Full brief"):
        for w, lines in items:
            st.markdown(f"**{w}:** " + " ".join(lines))


def links_line(st, links: dict[str, str]) -> None:
    st.caption(" · ".join(f"[{k}]({v})" for k, v in links.items()) + " · " + FREE_TIER)


def use_cards(st, pages: list[Page], title: str = "Use it") -> None:
    """One card per page (skipping the overview): what the viewer can do there, and a button to go."""
    st.html(f'<div class="f-label" style="margin-top:.9rem">{title}</div>')
    rest = pages[1:]
    per_row = 4 if len(rest) % 4 == 0 or len(rest) > 6 else 3
    for i in range(0, len(rest), per_row):
        cols = st.columns(per_row)
        for c, p in zip(cols, rest[i:i + per_row]):
            with c.container(border=True):
                st.markdown(f"**{p.label}**")
                st.caption(p.what)
                st.button("Open", icon=p.icon, key=f"go_{p.key}", on_click=_go, args=(p.key,))


def header(st, page: Page, question: str, answer: str, eyebrow: str = "") -> None:
    """Every page opens with what it shows and the business question it answers (answer from the data, not prose)."""
    st.html((f'<p class="f-eyebrow">{eyebrow}</p>' if eyebrow else "") + f'<h2 class="f-ptitle">{html.escape(page.label)}</h2>'
            f'<div class="f-context">{md_html(page.what)}</div>'
            f'<div class="f-biz"><div class="f-w">Business question</div><div class="q">{md_html(question)}</div>'
            f'<div class="a">{md_html(answer)}</div></div>')
