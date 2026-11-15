"""
LLM providers for the chatbot. Each one has the same single method:

    generate_json(system, user, schema, max_tokens) -> dict

so the pipeline in chatbot.py doesn't care which provider is behind it.

Choosing a provider (make_llm):
- LLM_PROVIDER=gemini or LLM_PROVIDER=claude forces one.
- Otherwise: Gemini if GEMINI_API_KEY (or GOOGLE_API_KEY) is set, else Claude if
  ANTHROPIC_API_KEY is set, else None (the bot runs offline).

Model overrides: GEMINI_MODEL (default below), CLAUDE_MODEL.
"""

import json
import os
from typing import Dict, Optional

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    """The provider answered, but not with usable JSON (blocked, truncated, empty)."""


class GeminiLLM:
    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        from google import genai  # pip install google-genai
        from google.genai import types

        self._types = types
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def generate_json(self, system: str, user: str, schema: Dict, max_tokens: int) -> Dict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        candidate = response.candidates[0] if response.candidates else None
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None and getattr(reason, "name", str(reason)) == "MAX_TOKENS":
            raise LLMError("response hit max_output_tokens before finishing")
        if not response.text:
            blocked = getattr(response, "prompt_feedback", None)
            raise LLMError(f"empty response (finish_reason={reason}, prompt_feedback={blocked})")
        return json.loads(response.text)


class ClaudeLLM:
    name = "claude"

    # Server-side fallback: a declined request is retried on Anthropic's
    # recommended fallback model inside the same call.
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: Optional[str] = None):
        import anthropic  # pip install anthropic

        self.model = model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def generate_json(self, system: str, user: str, schema: Dict, max_tokens: int) -> Dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            betas=[self.FALLBACK_BETA],
            fallbacks="default",
            system=system,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise LLMError("request was declined (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before finishing")
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


PROVIDERS = {"gemini": GeminiLLM, "claude": ClaudeLLM}


def detect_provider() -> Optional[str]:
    """Returns the provider name to use from the environment, or None if no key is set."""
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced:
        return forced
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def make_llm(provider: Optional[str] = None):
    """Builds the chosen provider, or returns None (with a printed reason) so the bot runs offline."""
    provider = provider or detect_provider()
    if provider is None:
        print("No LLM API key found (GEMINI_API_KEY or ANTHROPIC_API_KEY); running offline.")
        return None
    if provider not in PROVIDERS:
        print(f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(PROVIDERS)}. Running offline.")
        return None
    try:
        return PROVIDERS[provider]()
    except ImportError:
        package = "google-genai" if provider == "gemini" else "anthropic"
        print(f"{package} isn't installed (pip install -r requirements.txt); running offline.")
    except Exception as exc:  # e.g. the SDK rejects a missing key at construction
        print(f"Couldn't start {provider}: {type(exc).__name__}: {exc}. Running offline.")
    return None
