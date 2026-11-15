"""
TunedIn chatbot: understand mood -> retrieve songs -> generate a grounded reply.

    from src.chatbot import TunedInBot
    bot = TunedInBot()
    print(bot.answer("rough day, I just want to zone out").text)

Reliability rules:
- Crisis messages get a support response before anything else (no API call).
- If the mood model isn't confident, Gemini (src/llm.py) reads the mood; if that
  fails, the local models' best guess is used.
- Gemini writes a short paragraph but can only pick songs from the retrieved set
  (song_ids is an enum in the output schema), and every catalog title the
  paragraph mentions must be one of its picks. If Gemini fails, the bot writes
  the paragraph itself from the ranked list.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from src.recommender import load_songs, recommend_songs
    from src.catalog import contains_phrase, find_named_genre
    from src.mood_model import (EMOTIONS, KeywordMoodModel, MoodPrediction,
                                load_mood_model, profile_for_emotion)
    from src.llm import make_llm
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs
    from catalog import contains_phrase, find_named_genre
    from mood_model import (EMOTIONS, KeywordMoodModel, MoodPrediction,
                            load_mood_model, profile_for_emotion)
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
    "You are TunedIn, a warm, concise music companion. Choose the genre and 2 or 3 songs "
    "from <retrieved_songs> that best fit the user's mood. Then write ONE short paragraph "
    "(3-4 sentences, plain prose: no lists, headings, or markdown) that acknowledges how "
    "they feel and explains why that genre and each song fits, using the songs' mood, "
    "energy, and positivity (valence). Mention each chosen song by its exact title and "
    "artist. Put the chosen songs' song_id values in song_ids, in the order you mention "
    "them. Never mention any song that is not in <retrieved_songs>."
)


def is_crisis(message: str) -> bool:
    text = message.lower()
    return any(re.search(p, text) for p in CRISIS_PATTERNS)


@dataclass
class BotReply:
    text: str
    picks: List[Dict] = field(default_factory=list)       # recommended song dicts, in the order the reply mentions them
    emotion: Optional[str] = None
    mood_source: Optional[str] = None                     # keyword / fine-tuned / gemini
    profile: Optional[Dict] = None
    retrieved_ids: List[int] = field(default_factory=list)
    reply_source: str = "none"                            # gemini / fallback / crisis / empty
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
        self.moods = sorted({s["mood_family"] for s in self.songs})
        self.genres = sorted({s["genre"].lower() for s in self.songs})          # subgenres
        self.genre_families = sorted({s["genre_family"] for s in self.songs})   # pop, rock, ...
        self.mood_model = mood_model or load_mood_model(songs=self.songs)
        self.backup_model = KeywordMoodModel(self.songs)
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
        """Returns a catalog genre the user names ("some blues"), else ""."""
        return find_named_genre(message, self.genres)

    def _llm_understand(self, message: str) -> MoodPrediction:
        schema = {
            "type": "object",
            "properties": {
                "emotion": {"type": "string", "enum": EMOTIONS},
                "target_energy": {"type": "number"},
                "target_valence": {"type": "number"},
                "likes_acoustic": {"type": "boolean"},
                "favorite_genre": {"type": "string", "enum": self.genre_families + [""]},
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
        # IDs go out as strings: string enums are the most portable schema form.
        ids = [str(song["id"]) for song, _, _ in hits]
        schema = {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "enum": sorted({s["genre"] for s, _, _ in hits})},
                "song_ids": {"type": "array", "items": {"type": "string", "enum": ids}},
                "paragraph": {"type": "string"},
            },
            "required": ["genre", "song_ids", "paragraph"],
            "additionalProperties": False,
        }
        context = "\n".join(
            f"- song_id {s['id']}: \"{s['title']}\" by {s['artist']} | {s['genre']} / {s['mood']} | "
            f"energy {s['energy']}, valence {s['valence']}, acousticness {s['acousticness']}"
            for s, _, _ in hits
        )
        prompt = f"<retrieved_songs>\n{context}\n</retrieved_songs>\n\n<user_message>\n{message}\n</user_message>"
        data = self.llm.generate_json(GENERATE_SYSTEM, prompt, schema, max_tokens=4096)
        data["song_ids"] = list(dict.fromkeys(int(i) for i in data["song_ids"]))  # dedupe, keep order
        return data

    def mentioned_titles(self, text: str) -> List[int]:
        """
        IDs of catalog songs the text refers to by title. Whole-word, case- and
        accent-insensitive ("Senorita" finds "Señorita"). Many titles are common
        words ("Perfect", "Stay", "Closer"), so a one-word title only counts
        when it's in quotes or its artist is also named.
        """
        return [s["id"] for s in self.songs if _mentions_song(text, s)]

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
                pick_ids = data["song_ids"][:3]
                mentioned = self.mentioned_titles(data["paragraph"])
                reply.grounded = (
                    all(i in reply.retrieved_ids for i in pick_ids)
                    and all(i in pick_ids for i in mentioned)
                )
                if reply.grounded and pick_ids and data["paragraph"].strip():
                    reply.picks = [dict(self.by_id[i]) for i in pick_ids]
                    reply.text = data["paragraph"].strip()
                    reply.reply_source = self.llm.name
                    return reply
                errors.append("generate: reply named songs outside the retrieved set"
                              if not reply.grounded else "generate: empty reply")
            except Exception as exc:  # any API/parse failure -> template paragraph
                errors.append(f"generate: {type(exc).__name__}: {exc}")

        reply.picks = [dict(s) for s, _, _ in hits[:3]]
        reply.text = _fallback_paragraph(prediction.profile, reply.picks)
        reply.reply_source = "fallback"
        return reply


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _primary_artist(artist: str) -> str:
    """ "Post Malone & Swae Lee" -> "Post Malone"; "Macklemore & Ryan Lewis ft. X" -> "Macklemore". """
    return re.split(r"\s*(?:&|,| ft\.? | feat\.? | featuring | x | with )\s*", artist, maxsplit=1, flags=re.I)[0]


def _mentions_song(text: str, song: Dict) -> bool:
    title = song["title"]
    if not contains_phrase(text, title):
        return False
    if len(title.split()) > 1:
        return True
    quoted = any(contains_phrase(text, q + title + q) for q in ('"', "'"))
    return quoted or contains_phrase(text, _primary_artist(song["artist"]))


def _describe(song: Dict) -> str:
    energy = "high-energy" if song["energy"] >= 0.7 else "mellow" if song["energy"] <= 0.4 else "mid-tempo"
    tone = "bright" if song["valence"] >= 0.65 else "somber" if song["valence"] <= 0.35 else "bittersweet"
    return f"{energy}, {tone} {song['genre']} track with a {song['mood']} feel"


def _fallback_paragraph(profile: Dict, picks: List[Dict]) -> str:
    """Offline reply: the same short-paragraph shape, written from the ranked songs."""
    mood = profile["favorite_mood"]
    genre = profile.get("favorite_genre") or picks[0]["genre"]
    first, rest = picks[0], picks[1:]
    text = (f"Since you're in a {mood} mood, {genre} is a good place to start. "
            f"\"{first['title']}\" by {first['artist']} is a {_describe(first)}, "
            f"so it sits close to how you're feeling.")
    if rest:
        others = " and ".join(f"\"{s['title']}\" by {s['artist']}" for s in rest)
        text += (f" {others} keep a similar energy and positivity, "
                 f"so they fit the same {mood} space.")
    return text


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
