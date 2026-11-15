"""Gemini setup tests. No API calls: only checks when a client gets built."""

from src.llm import has_gemini_key, make_llm

KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _clear(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def test_no_key_means_offline(monkeypatch):
    _clear(monkeypatch)
    assert not has_gemini_key()
    assert make_llm() is None


def test_google_api_key_also_works(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    assert has_gemini_key()


def test_gemini_client_builds_with_a_key(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    llm = make_llm()
    assert llm is not None and llm.name == "gemini"
