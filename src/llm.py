"""
Gemini LLM for the chatbot. One method:

    generate_json(system, user, schema, max_tokens) -> dict

Setup: set GEMINI_API_KEY (or GOOGLE_API_KEY). Optional: GEMINI_MODEL to
override the default model below. With no key, make_llm() returns None and the
chatbot runs offline (local mood model + ranked-list reply).
"""

import json
import os
from typing import Dict, Optional

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class LLMError(RuntimeError):
    """Gemini answered, but not with usable JSON (blocked, truncated, empty)."""


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


def has_gemini_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def make_llm():
    """Returns a GeminiLLM, or None (with a printed reason) so the bot runs offline."""
    if not has_gemini_key():
        print("No GEMINI_API_KEY found; running offline.")
        return None
    try:
        return GeminiLLM()
    except ImportError:
        print("google-genai isn't installed (pip install -r requirements.txt); running offline.")
    except Exception as exc:  # e.g. the SDK rejects the key format at construction
        print(f"Couldn't start Gemini: {type(exc).__name__}: {exc}. Running offline.")
    return None
