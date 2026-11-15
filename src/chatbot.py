"""
TunedIn chatbot: understand mood -> retrieve songs -> generate a grounded reply.

    from src.chatbot import TunedInBot
    bot = TunedInBot()
    print(bot.answer("rough day, I just want to zone out").text)

Reliability rules:
- Crisis messages get a support response before anything else (no API call).
- If the mood model isn't confident, the LLM (Gemini or Claude, see src/llm.py)
  reads the mood; if that fails, the local models' best guess is used.
- The LLM can only pick songs from the retrieved set (song_id is an enum in the
  output schema). If the LLM fails, the bot returns the ranked list on its own.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from src.recommender import load_songs, recommend_songs
    from src.mood_model import EMOTIONS, KeywordMoodModel, MoodPrediction, load_mood_model, profile_for_emotion
    from src.llm import make_llm
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs
    from mood_model import EMOTIONS, KeywordMoodModel, MoodPrediction, load_mood_model, profile_for_emotion
    from llm import make_llm

SONGS_CSV = Path(__file__).resolve().parent.parent / "data" / "songs.csv"

CRISIS_PATTERNS = [
    r"kill myself", r"suicid", r"end(ing)? it all", r"want to die", r"wanna die",
    r"don'?t want to (be here|live|exist)", r"self[- ]?harm", r"hurt myself",
    r"no reason to live", r"better off without me",
]

CRISIS_REPLY = (
    "I'm really sorry you're feeling this way. You don't have to go through it alone. "
    "If you're in the US, you can call or text 988 (Suicide & Crisis Lifeline) any time. "
    "Outside the US, findahelpline.com lists free, confidential lines in your country. "
    "If you're in immediate danger, please contact local emergency services. "
    "I'm here for music whenever you want it, but please reach out to someone first."
)

EMPTY_REPLY = "Tell me how you're feeling, or what kind of day it's been, and I'll find songs for it."

UNDERSTAND_SYSTEM = (
    "You read a user's message and describe their emotional state as a music profile. "
    "Pick the closest emotion label. target_energy and target_valence are 0-1: energy is "
    "calm (0) to excited (1); valence is sad (0) to happy (1). Match the music to the "
    "feeling they describe unless they ask for a different vibe (e.g. 'cheer me up' "
    "means higher valence). Set favorite_genre only if they name a genre, else empty."
)

GENERATE_SYSTEM = (
    "You are TunedIn, a warm, concise music companion. Recommend songs ONLY from "
    "<retrieved_songs>, using their song_id. Pick 2 or 3 songs that best fit the user's "
    "mood, name the main genre you're recommending, and give one short sentence per song "
    "on why it fits how they feel. Write intro as one or two friendly sentences that "
    "acknowledge their mood. Never mention songs that are not in the list."
)


def is_crisis(message: str) -> bool:
    text = message.lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


@dataclass
class BotReply:
    text: str
    picks: List[Dict] = field(default_factory=list)       # song dicts + "why"
    emotion: Optional[str] = None
    mood_source: Optional[str] = None                     # keyword / fine-tuned / gemini / claude
    profile: Optional[Dict] = None
    retrieved_ids: List[int] = field(default_factory=list)
    reply_source: str = "none"                            # gemini / claude / fallback / crisis / empty
    grounded: bool = True                                 # every pick was in the retrieved set
    errors: List[str] = field(default_factory=list)
    latency_s: float = 0.0


class TunedInBot:
    def __init__(
        self,
        songs: Optional[List[Dict]] = None,
        mood_model=None,
        llm=None,
        use_llm: bool = True,
        confidence_threshold: float = 0.6,
        k: int = 5,
    ):
        self.songs = songs if songs is not None else load_songs(str(SONGS_CSV))
        self.by_id = {s["id"]: s for s in self.songs}
        self.moods = sorted({s["mood"] for s in self.songs})
        self.genres = sorted({s["genre"] for s in self.songs})
        self.mood_model = mood_model or load_mood_model()
        self.backup_model = KeywordMoodModel()
        self.confidence_threshold = confidence_threshold
        self.k = k
        self.llm = llm
        if use_llm and llm is None:
            self.llm = make_llm()

    # -- 1. Understand ------------------------------------------------------

    def understand(self, message: str, errors: List[str]) -> MoodPrediction:
        guess = self.mood_model.predict(message)
        if guess.confidence >= self.confidence_threshold:
            return self._with_genre(guess, message)
        if self.llm is not None:
            try:
                return self._llm_understand(message)
            except Exception as exc:  # any API/parse failure -> fall through to local models
                errors.append(f"understand: {type(exc).__name__}: {exc}")
        # Primary model is unsure and the LLM is unavailable: if the keyword
        # baseline found any mood words, trust it over a low-confidence guess.
        if guess.source != self.backup_model.name:
            backup = self.backup_model.predict(message)
            if backup.confidence > 0:
                guess = backup
        return self._with_genre(guess, message)

    def _with_genre(self, guess: MoodPrediction, message: str) -> MoodPrediction:
        named = self.named_genre(message)
        if named:
            guess.profile["favorite_genre"] = named
        return guess

    def named_genre(self, message: str) -> str:
        """Returns a catalog genre the user names ("some blues"), longest match first, else ""."""
        text = message.lower()
        for genre in sorted(self.genres, key=len, reverse=True):  # "indie pop" before "pop"
            if re.search(rf"(?<!\w){re.escape(genre)}(?!\w)", text):
                return genre
        return ""

    def _llm_understand(self, message: str) -> MoodPrediction:
        schema = {
            "type": "object",
            "properties": {
                "emotion": {"type": "string", "enum": EMOTIONS},
                "target_energy": {"type": "number"},
                "target_valence": {"type": "number"},
                "likes_acoustic": {"type": "boolean"},
                "favorite_genre": {"type": "string", "enum": self.genres + [""]},
            },
            "required": ["emotion", "target_energy", "target_valence",
                         "likes_acoustic", "favorite_genre"],
            "additionalProperties": False,
        }
        data = self.llm.generate_json(UNDERSTAND_SYSTEM, message, schema, max_tokens=2048)
        profile = profile_for_emotion(data["emotion"])
        profile.update(
            target_energy=_clamp(data["target_energy"]),
            target_valence=_clamp(data["target_valence"]),
            likes_acoustic=bool(data["likes_acoustic"]),
            favorite_genre=data["favorite_genre"],
        )
        return MoodPrediction(data["emotion"], 1.0, profile, self.llm.name)

    # -- 2. Retrieve --------------------------------------------------------

    def retrieve(self, profile: Dict):
        return recommend_songs(profile, self.songs, k=self.k)

    # -- 3. Generate --------------------------------------------------------

    def _llm_generate(self, message: str, hits) -> Dict:
        # IDs go out as strings: string enums work the same on every provider.
        ids = [str(song["id"]) for song, _, _ in hits]
        schema = {
            "type": "object",
            "properties": {
                "intro": {"type": "string"},
                "genre": {"type": "string", "enum": sorted({s["genre"] for s, _, _ in hits})},
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "song_id": {"type": "string", "enum": ids},
                            "why": {"type": "string"},
                        },
                        "required": ["song_id", "why"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["intro", "genre", "picks"],
            "additionalProperties": False,
        }
        context = "\n".join(
            f"- song_id {s['id']}: \"{s['title']}\" by {s['artist']} | {s['genre']} / {s['mood']} | "
            f"energy {s['energy']}, valence {s['valence']}, acousticness {s['acousticness']}"
            for s, _, _ in hits
        )
        prompt = f"<retrieved_songs>\n{context}\n</retrieved_songs>\n\n<user_message>\n{message}\n</user_message>"
        data = self.llm.generate_json(GENERATE_SYSTEM, prompt, schema, max_tokens=4096)
        for p in data["picks"]:
            p["song_id"] = int(p["song_id"])
        return data

    # -- Pipeline -----------------------------------------------------------

    def answer(self, message: str) -> BotReply:
        start = time.perf_counter()
        reply = self._answer(message or "")
        reply.latency_s = time.perf_counter() - start
        return reply

    def _answer(self, message: str) -> BotReply:
        if not message.strip():
            return BotReply(text=EMPTY_REPLY, reply_source="empty")
        if is_crisis(message):
            return BotReply(text=CRISIS_REPLY, reply_source="crisis")

        errors: List[str] = []
        prediction = self.understand(message, errors)
        hits = self.retrieve(prediction.profile)
        reply = BotReply(
            text="",
            emotion=prediction.emotion,
            mood_source=prediction.source,
            profile=prediction.profile,
            retrieved_ids=[s["id"] for s, _, _ in hits],
            errors=errors,
        )

        if self.llm is not None:
            try:
                data = self._llm_generate(message, hits)
                picks = _dedupe_picks(data["picks"])[:3]
                reply.grounded = all(p["song_id"] in reply.retrieved_ids for p in picks)
                picks = [p for p in picks if p["song_id"] in reply.retrieved_ids]
                if picks:
                    reply.picks = [{**self.by_id[p["song_id"]], "why": p["why"]} for p in picks]
                    reply.text = _render(data["intro"], data["genre"], reply.picks)
                    reply.reply_source = self.llm.name
                    return reply
                errors.append("generate: no valid picks returned")
            except Exception as exc:  # any API/parse failure -> plain ranked list
                errors.append(f"generate: {type(exc).__name__}: {exc}")

        reply.picks = [
            {**s, "why": f"{s['mood']} feel, energy {s['energy']:.2f}, positivity {s['valence']:.2f}"}
            for s, _, _ in hits[:3]
        ]
        reply.text = _render(
            f"Here's what fits a {prediction.profile['favorite_mood']} mood right now.",
            reply.picks[0]["genre"],
            reply.picks,
        )
        reply.reply_source = "fallback"
        return reply


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dedupe_picks(picks: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for p in picks:
        if p["song_id"] not in seen:
            seen.add(p["song_id"])
            out.append(p)
    return out


def _render(intro: str, genre: str, picks: List[Dict]) -> str:
    lines = [intro, "", f"Genre to try: {genre}", ""]
    for i, p in enumerate(picks, start=1):
        lines.append(f"{i}. {p['title']} by {p['artist']} ({p['genre']}): {p['why']}")
    return "\n".join(lines)


if __name__ == "__main__":
    bot = TunedInBot()
    print("TunedIn: tell me how you're feeling (Ctrl+C to quit).")
    while True:
        try:
            msg = input("\nyou> ")
        except (KeyboardInterrupt, EOFError):
            break
        r = bot.answer(msg)
        print(f"\n{r.text}\n\n[mood: {r.emotion} via {r.mood_source} | reply: {r.reply_source} | {r.latency_s:.1f}s]")
        for e in r.errors:
            print(f"[warning] {e}")
