"""Record Ask AI answers for the app's replay chat (R26), with retrieved sources, number check, cost and latency.

    set -a; . ./.env; set +a; python scripts/record_app_replays.py      # needs OPENAI_API_KEY; prints cost, never the key
Writes src/ui/replays_ask_ai.json from committed benchmark files only.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "ui"))
import ask_ai as K  # noqa: E402


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    results = K.load_results()
    runs = []
    for q in K.QUESTIONS:
        r = K.live(q, None, results)
        runs.append(r)
        print(f"{r['latency_ms']:>7.0f} ms  ${r['cost_usd']:.6f}  grounded={r['grounded']}  {q[:60]}  {r['ungrounded_numbers'] or ''}")
    lat = sorted(r["latency_ms"] for r in runs)
    out = {"recorded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"), "model": K.MODEL,  # noqa: UP017 (3.8+)
           "source": "the benchmark's committed results (fair rolling benchmark first)", "runs": runs,
           "summary": {"questions": len(runs), "grounded": sum(r["grounded"] for r in runs),
                       "cost_usd_total": round(sum(r["cost_usd"] for r in runs), 6),
                       "cost_per_1000_usd": round(1000 * sum(r["cost_usd"] for r in runs) / len(runs), 4),
                       "latency_ms_p50": lat[len(lat) // 2], "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))]}}
    (ROOT / "src" / "ui" / "replays_ask_ai.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
