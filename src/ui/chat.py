"""Chat tab for the Streamlit companion apps (R26): replay by default, live only with the viewer's own OpenAI key.

Same file in every portfolio app. The viewer's key lives in their session only: never written to disk, logged, or sent anywhere
but OpenAI. Live questions are capped per session.
T (the project's adapter) gives MODEL and either `live(question, key) -> run` or SYSTEM, TOOLS and PRICES for the shared loop.
A run may list `images` (paths relative to the replay file) that the agent drew.
"""
from __future__ import annotations

from pathlib import Path

import agent as A

LIVE_LIMIT = 10                  # live questions per session
MATCH_FLOOR = 0.35               # below this, no recorded answer is close enough to show


def _show(st, run: dict, note: str | None = None, base: Path | None = None) -> None:
    with st.chat_message("user"):
        st.markdown(run["question"])
    with st.chat_message("assistant"):
        if note:
            st.caption(note)
        st.markdown(run["answer"] or "_No answer._")
        for img in run.get("images", []):
            p = Path(img) if Path(img).is_absolute() or base is None else base / img
            if p.exists():
                st.image(str(p))
        with st.expander("How it answered: tools, number check, cost"):
            if run["tool_calls"]:
                for i, c in enumerate(run["tool_calls"], 1):
                    args = ", ".join(f"{k}={v}" for k, v in c["arguments"].items())
                    st.markdown(f"{i}. `{c['name']}({args})`" + ("" if c["ok"] else " (returned an error)"))
            else:
                st.markdown("No tools: the question is outside what this agent does.")
            cited = [r for r in run.get("tool_results", []) if isinstance(r, dict) and r.get("cited") and r.get("text")]
            if cited:
                st.markdown("**Cited sources:**")
                for r in cited:
                    st.markdown(f"- `{r.get('id', '')}` {r.get('neighbourhood', '')}: \"{r['text'][:240]}…\"")
            check = ("every number in the answer is in a tool result or the question" if run["grounded"]
                     else f"numbers not found in the tool results: {', '.join(run['ungrounded_numbers'])}")
            st.markdown(f"- **Number check:** {check}\n- **Model:** {run['model']}, {run['steps']} round(s)\n"
                        f"- **Cost:** ${run['cost_usd']:.5f} ({run['tokens']['input']:,} tokens in, {run['tokens']['output']:,} out)\n"
                        f"- **Time:** {run['latency_ms'] / 1000:.1f} s")


def render(st, replays_path: Path, T, prefix: str = "") -> None:
    """One chat surface. `prefix` keeps widget keys apart when a page has more than one."""
    rp = A.load_replays(replays_path)
    runs, s, base = rp["runs"], rp["summary"], Path(replays_path).resolve().parent
    source = rp.get("source", "the public API")
    mode = st.radio("Mode", ["Replay: recorded answers (free)", "Live: use your own OpenAI key"], horizontal=True, key=f"{prefix}chat_mode")
    if mode.startswith("Replay"):
        st.caption(f"Answers recorded from the real agent ({rp['model']}) on {source}, {rp['recorded_utc']} UTC: "
                   f"{s['grounded']} of {s['questions']} pass the number check; about ${s['cost_per_1000_usd']:.2f} per 1,000 questions; "
                   f"p95 {s['latency_ms_p95'] / 1000:.1f} s. No model runs here.")
        pick = st.selectbox("Pick a recorded question", [r["question"] for r in runs], index=None,
                            placeholder="Choose a question the agent answered", key=f"{prefix}replay_pick")
        typed = st.chat_input("Or type a question; the closest recorded one is shown", key=f"{prefix}replay_input")
        if typed:
            best, score = A.closest(typed, runs)
            if score < MATCH_FLOOR:
                with st.chat_message("user"):
                    st.markdown(typed)
                st.info("No recorded answer is close to that question. Pick one above, or switch to Live and use your own OpenAI key.")
            else:
                _show(st, best, f"Closest recorded question (match {score:.2f}): \"{best['question']}\"", base)
        elif pick:
            _show(st, next(r for r in runs if r["question"] == pick), None, base)
        return

    st.markdown(f"Ask anything in scope; the agent runs live and answers with **{T.MODEL}** "
                f"(about ${s['cost_per_1000_usd'] / 1000:.4f} a question on your account). Up to {LIVE_LIMIT} questions per session.")
    st.caption("Your key stays in this browser session only: it is used to call OpenAI from this app's server and is never stored, "
               "logged or shared. Closing the tab clears it. Use a key with a spending limit.")
    key = st.text_input("Your OpenAI API key", type="password", key=f"{prefix}byo_key", placeholder="sk-...")
    history = st.session_state.setdefault(f"{prefix}live_history", [])
    q = st.chat_input("Ask the agent" if key else "Enter your key above to ask", key=f"{prefix}live_input", disabled=not key)
    if q and key:
        if len(history) >= LIVE_LIMIT:
            st.warning(f"This session has used its {LIVE_LIMIT} live questions. Reload the page to start again.")
        else:
            try:
                with st.spinner("The agent is working..."):
                    if hasattr(T, "live"):
                        history.append(T.live(q, key))
                    else:
                        from openai import OpenAI
                        history.append(A.run(q, OpenAI(api_key=key, timeout=30, max_retries=1), T.TOOLS, T.SYSTEM, T.MODEL, T.PRICES))
            except Exception as e:  # noqa: BLE001 - bad key, quota, network: say what happened, never echo the key
                st.error(f"The live call failed: {A.redact(str(e), key)[:300]}")
    for run in history:
        _show(st, run, None, base)
