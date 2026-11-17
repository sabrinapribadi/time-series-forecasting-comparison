"""Chat engine for the Streamlit companion apps (R26): replay recorded agent runs by default; live runs only with the viewer's own key.

The same file is used by every portfolio app; each project supplies its tools in app/agent_tools.py (SYSTEM, TOOLS, MODEL, PRICES).
- Replay: answers recorded from the real agent on the public API (scripts/record_agent_replays.py), matched to the viewer's question.
- Live: the viewer's OpenAI key, held in their session only; it is never written to disk, logged or sent anywhere but OpenAI.
Every answer is checked: each number in it must appear in the question or a tool result (within rounding).
"""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path

_NUM = re.compile(r"(?<![\w.])([-−]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*%|[kKM]\b)?")   # "2 m" is metres, "2.1M" millions
STOP = {"the", "a", "an", "of", "to", "in", "is", "and", "for", "what", "how", "if", "we", "our", "do", "does", "should", "at", "on", "it", "be"}


# ---------------------------------------------------------------- grounding (same rule as the projects' agent graders)
def _value(m: re.Match) -> float:
    v = float(m.group(2).replace(",", "") + (f".{m.group(3)}" if m.group(3) else ""))
    return -v if m.group(1) else v


def _leaves(x, out: list) -> None:
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, str):
        s = x.strip()
        if s[:1] in "{[":                                   # a tool that returns JSON as text: read its values, not its escapes
            try:
                _leaves(json.loads(s), out)
                return
            except ValueError:
                pass
        out += [_value(m) for m in _NUM.finditer(x)]
    elif isinstance(x, dict):
        for k, v in x.items():
            out += [float(n) for n in re.findall(r"(?<=_)\d+(?=_|$)", str(k))]   # chance_below_25_mbps -> 25
            out += [_value(m) for m in _NUM.finditer(str(k))]                   # "4.0–4.4 (good)" -> 4.0, 4.4
            _leaves(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _leaves(v, out)


def ungrounded_numbers(text: str, sources: list) -> list[str]:
    """Numbers in the answer that are not in the question or any tool result (percent and rounding allowed)."""
    vals: list[float] = []
    _leaves(sources, vals)
    pool = vals + [v * 100 for v in vals if abs(v) <= 10.0] + [v / 100 for v in vals]   # 0.93 -> 93%; a 1.85 gain -> 185%
    bad = []
    for m in _NUM.finditer(text):
        dec, unit = m.group(3) or "", (m.group(4) or "").strip().lower()
        scale = {"k": 1e3, "m": 1e6}.get(unit, 1.0)
        v, tol = _value(m) * scale, 0.5 * 10 ** -len(dec) * scale
        if abs(v) <= 10 and float(v).is_integer():
            continue
        if not any(abs(v - s) <= max(tol, 1e-9) or (s and abs(v - s) / abs(s) <= 0.005) for s in pool):
            bad.append(m.group(0).strip())
    return bad


# ---------------------------------------------------------------- replay
def load_replays(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.\-]+", s.lower()) if w not in STOP}


def closest(question: str, runs: list[dict]) -> tuple[dict, float]:
    """The recorded run whose question is closest (word overlap blended with character similarity), and the score 0-1."""
    q = _words(question)

    def score(r: dict) -> float:
        w = _words(r["question"])
        jac = len(q & w) / max(1, len(q | w))
        return 0.6 * jac + 0.4 * difflib.SequenceMatcher(None, question.lower(), r["question"].lower()).ratio()
    best = max(runs, key=score)
    return best, round(score(best), 2)


# ---------------------------------------------------------------- live
def redact(text: str, key: str | None) -> str:
    """Never echo a key back, even inside an error message."""
    if key:
        text = text.replace(key, "[your key]")
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[key]", text)


def run(question: str, client, tools: dict, system: str, model: str, prices: dict, max_steps: int = 4,
        max_output_tokens: int = 400, reasoning_effort: str | None = "none") -> dict:
    """One question through the tool-using loop (OpenAI Responses API). `tools`: name -> {"schema": {...}, "fn": callable}."""
    t0 = time.perf_counter()
    items: list = [{"role": "user", "content": question}]
    calls, results = [], []
    it = ct = ot = 0
    text, steps = "", 0
    extra = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}
    schemas = [t["schema"] for t in tools.values()]
    for steps in range(1, max_steps + 1):
        last = steps == max_steps
        r = client.responses.create(model=model, instructions=system, input=items, max_output_tokens=max_output_tokens,
                                    tools=schemas, tool_choice="none" if last else "auto", **extra)
        u = getattr(r, "usage", None)
        it += int(getattr(u, "input_tokens", 0) or 0)
        ot += int(getattr(u, "output_tokens", 0) or 0)
        ct += int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0)
        fcalls = [o for o in (getattr(r, "output", None) or []) if getattr(o, "type", "") == "function_call"]
        if not fcalls:
            text = (getattr(r, "output_text", "") or "").strip()
            break
        items += list(r.output)
        for fc in fcalls:
            try:
                args = json.loads(fc.arguments or "{}")
                res = tools[fc.name]["fn"](**args) if fc.name in tools else {"error": f"unknown tool {fc.name}"}
            except Exception as e:  # noqa: BLE001 - bad arguments or an API error go back to the model as data
                args, res = {}, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            calls.append({"name": fc.name, "arguments": args, "ok": not (isinstance(res, dict) and "error" in res)})
            results.append(res)
            items.append({"type": "function_call_output", "call_id": fc.call_id, "output": json.dumps(res, default=str)})
    bad = ungrounded_numbers(text, [question, results])
    cost = ((it - ct) * prices["input"] + ct * prices["cached_input"] + ot * prices["output"]) / 1e6
    return {"question": question, "answer": text, "tool_calls": calls, "tool_results": results, "steps": steps,
            "grounded": not bad and bool(text), "ungrounded_numbers": bad, "model": model,
            "tokens": {"input": it, "cached_input": ct, "output": ot}, "cost_usd": round(cost, 8),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
