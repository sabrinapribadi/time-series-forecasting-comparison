"""Streamlit app (R26): first screen, Ask AI in replay, live mode with a stub, and no owner key anywhere; offline."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "ui"
sys.path.insert(0, str(UI))
pytest.importorskip("streamlit")
import agent as A  # noqa: E402
import ask_ai  # noqa: E402
import showcase  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(UI / "dashboard.py")


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


PAGES = ["overview", "explore", "fair", "gallery", "inspector", "tests", "original", "rl", "agent"]


def app(page="overview"):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception
    return at


def test_first_screen_reports_the_fair_benchmark():
    at = app()
    h = showcase.headline()
    assert [m.value for m in at.metric][:2] == [f"{h['mase_f']:.3f}", f"{h['mase_t']:.3f}"]
    assert any("12 hours" in c.value for c in at.caption)
    body = " ".join(e.proto.body for e in at.get("html"))
    for w in ["What", "Why", "Who", "Where", "When it fails", "How"]:          # the 5W1H brief (R22)
        assert f'class="f-w">{w}<' in body, w
    assert [b.key for b in at.sidebar.button] == [f"nav_{k}" for k in PAGES]


def test_every_page_opens_with_its_business_question():
    for key in PAGES[1:]:
        at = app(key)
        assert "Business question" in " ".join(e.proto.body for e in at.get("html")), key


def test_knowledge_leads_with_the_fair_benchmark():
    ch = ask_ai.knowledge(ask_ai.load_results())
    assert ch[0]["id"] == "fair_benchmark" and "1. Chronos-2 (foundation)" in ch[0]["text"]
    orig = next(c for c in ch if c["id"] == "original_benchmark")
    assert orig["text"].startswith("ORIGINAL") and "NOT like-for-like" in orig["text"]
    for q in ask_ai.QUESTIONS:                                    # retrieval always carries the fair result
        assert any(c["id"] == "fair_benchmark" for c in ask_ai.retrieve(q, ch))


def test_replays_recorded_grounded_and_keyless():
    p = UI / "replays_ask_ai.json"
    d = json.loads(p.read_text())
    assert [r["question"] for r in d["runs"]] == ask_ai.QUESTIONS and all(r["grounded"] for r in d["runs"])
    assert "sk-" not in p.read_text()


def test_ask_ai_replay_and_live_stub(monkeypatch):
    at = app("agent")
    assert at.radio(key="ask_chat_mode").value.startswith("Replay")
    run = json.loads((UI / "replays_ask_ai.json").read_text())["runs"][0]
    at.selectbox(key="ask_replay_pick").set_value(run["question"]).run()
    assert not at.exception and any(run["answer"][:40] in m.value for m in at.markdown)

    key = "sk" + "-" + "test-viewerkey-123456"                    # fake, built at runtime
    seen = {}

    def fake(question, k, results=None):
        seen["key"] = k
        return {**run, "question": question, "answer": "Live answer."}
    monkeypatch.setattr(ask_ai, "live", fake)
    at = app("agent")
    at.radio(key="ask_chat_mode").set_value("Live: use your own OpenAI key").run()
    at.text_input(key="ask_byo_key").input(key).run()
    at.chat_input(key="ask_live_input").set_value("Anything").run()
    assert seen["key"] == key and any("Live answer." in m.value for m in at.markdown)
    assert key not in " ".join(m.value for m in at.markdown)
    assert A.redact(f"bad key {key}", key) == "bad key [your key]"


def test_no_owner_key_is_read():
    src = (UI / "dashboard.py").read_text()
    assert "st.secrets" not in src and "OPENAI_API_KEY" not in src


def test_frame_is_the_shared_copy():
    """The app frame is identical in every portfolio app; change it everywhere at once (checksum of the shared copy)."""
    import hashlib
    assert hashlib.sha256((UI / "frame.py").read_bytes()).hexdigest()[:16] == "385e92329f20405a"
