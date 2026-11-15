"""Provider selection tests. No API calls: only checks which provider would be chosen."""

from src.llm import detect_provider, make_llm

KEYS = ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY")


def _clear(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def test_no_keys_means_offline(monkeypatch):
    _clear(monkeypatch)
    assert detect_provider() is None
    assert make_llm() is None


def test_gemini_key_selects_gemini_even_with_anthropic_key(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert detect_provider() == "gemini"


def test_anthropic_key_alone_selects_claude(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert detect_provider() == "claude"


def test_llm_provider_overrides_detection(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("LLM_PROVIDER", "Claude")
    assert detect_provider() == "claude"


def test_unknown_provider_runs_offline(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    assert make_llm() is None


def test_gemini_client_builds_with_a_key(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    llm = make_llm()
    assert llm is not None and llm.name == "gemini"
