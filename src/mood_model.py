"""
Mood understanding: turns a user's message into a profile the scorer understands.

Two models share one interface, predict(message) -> MoodPrediction:

- KeywordMoodModel: a hand-written keyword baseline. No dependencies, always
  available, and the thing every other model has to beat in the eval.
- FineTunedMoodModel: a DistilBERT classifier fine-tuned on GoEmotions by
  training/train_mood.py. Needs `torch` + `transformers` and the saved model in
  models/mood_classifier/.

Both predict a GoEmotions emotion label. EMOTION_PROFILES then maps that label
onto a mood from data/songs.csv plus target energy / valence / acoustic values.
Moods here are the canonical moods from src/catalog.py (each groups several
catalog moods, e.g. "sad" covers "heartbroken" and "despairing"). The keyword
model also picks a genre: the one the user names, otherwise the catalog's most
common genre family for the detected mood.
"""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from src.catalog import find_named_genre, normalize_song
except ModuleNotFoundError:
    from catalog import find_named_genre, normalize_song

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "mood_classifier"
SONGS_CSV = ROOT / "data" / "songs.csv"

# GoEmotions label -> (canonical mood, target_energy, target_valence, likes_acoustic)
# Every mood here must have at least one catalog song (tests/test_catalog.py). The numbers place each emotion
# on the energy (calm -> excited) and valence (sad -> happy) axes.
EMOTION_PROFILES: Dict[str, tuple] = {
    "admiration":     ("uplifting",  0.60, 0.80, False),
    "amusement":      ("playful",    0.70, 0.85, False),
    "anger":          ("angry",      0.95, 0.25, False),
    "annoyance":      ("angry",      0.80, 0.30, False),
    "approval":       ("confident",  0.65, 0.70, False),
    "caring":         ("romantic",   0.45, 0.70, True),
    "confusion":      ("moody",      0.50, 0.45, False),
    "curiosity":      ("chill",      0.45, 0.60, True),
    "desire":         ("romantic",   0.55, 0.70, False),
    "disappointment": ("sad",        0.30, 0.25, True),
    "disapproval":    ("moody",      0.60, 0.35, False),
    "disgust":        ("angry",      0.85, 0.25, False),
    "embarrassment":  ("moody",      0.40, 0.35, True),
    "excitement":     ("euphoric",   0.92, 0.85, False),
    "fear":           ("moody",      0.55, 0.30, False),
    "gratitude":      ("uplifting",  0.55, 0.85, True),
    "grief":          ("melancholy", 0.20, 0.20, True),
    "joy":            ("happy",      0.80, 0.85, False),
    "love":           ("romantic",   0.50, 0.75, True),
    "nervousness":    ("chill",      0.35, 0.50, True),
    "optimism":       ("uplifting",  0.65, 0.85, False),
    "pride":          ("confident",  0.75, 0.75, False),
    "realization":    ("chill",      0.45, 0.60, True),
    "relief":         ("chill",      0.35, 0.70, True),
    "remorse":        ("melancholy", 0.25, 0.25, True),
    "sadness":        ("sad",        0.30, 0.20, True),
    "surprise":       ("energetic",  0.80, 0.70, False),
    "neutral":        ("chill",      0.40, 0.55, True),
}

EMOTIONS = sorted(EMOTION_PROFILES)


@dataclass
class MoodPrediction:
    emotion: str          # GoEmotions label, e.g. "grief"
    confidence: float     # 0-1; how sure the model is
    profile: Dict         # keys match recommender.UserProfile
    source: str           # which model produced it ("keyword", "fine-tuned", "gemini")


def profile_for_emotion(emotion: str) -> Dict:
    """Builds a scorer profile from a GoEmotions label (unknown labels -> neutral)."""
    mood, energy, valence, acoustic = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    return {
        "favorite_genre": "",
        "favorite_mood": mood,
        "target_energy": energy,
        "target_valence": valence,
        "likes_acoustic": acoustic,
    }


# ---------------------------------------------------------------------------
# Keyword baseline
# ---------------------------------------------------------------------------

KEYWORDS: Dict[str, list] = {
    "grief":       ["passed away", "died", "funeral", "lost my", "grieving", "miss them", "miss her", "miss him"],
    "sadness":     ["sad", "down", "depressed", "cry", "crying", "lonely", "heartbroken", "broke up", "blue", "upset", "sick"],
    "anger":       ["angry", "furious", "mad", "pissed", "rage", "hate", "livid"],
    "annoyance":   ["annoyed", "irritated", "frustrated", "ugh", "fed up"],
    "nervousness": ["nervous", "anxious", "stressed", "stress", "worried", "panic", "overwhelmed", "exam"],
    "fear":        ["scared", "afraid", "terrified", "fear"],
    "excitement":  ["excited", "hyped", "pumped", "let's go", "lets go", "can't wait", "party", "!!!"],
    "joy":         ["happy", "great day", "amazing", "awesome", "good mood", "glad", "wonderful", "sunny"],
    "love":        ["love", "crush", "date night", "partner", "romantic", "anniversary"],
    "relief":      ["relieved", "finally done", "finished", "over with", "relax", "unwind", "zone out", "chill"],
    "pride":       ["proud", "nailed it", "crushed it", "confident", "got the job", "promoted"],
    "optimism":    ["hopeful", "optimistic", "fresh start", "new beginning", "looking forward"],
    "gratitude":   ["grateful", "thankful", "blessed"],
    "curiosity":   ["focus", "study", "studying", "concentrate", "work on", "essay", "coding", "homework"],
    "amusement":   ["funny", "lol", "haha", "silly", "goofy"],
    "remorse":     ["regret", "my fault", "guilty", "sorry i"],
    "disappointment": ["disappointed", "let down", "didn't get", "failed", "rejected"],
}


def _contains(text: str, keyword: str) -> bool:
    """Whole-word match, so "mad" doesn't fire on "made". Punctuation keywords like "!!!" match anywhere."""
    if not keyword[0].isalnum():
        return keyword in text
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None


MIN_SONGS_FOR_DEFAULT_GENRE = 2


def genre_for_each_mood(songs: List[Dict]) -> Dict[str, str]:
    """
    Most common genre family per mood family (ties go to the song listed first).
    A mood only gets a default genre if at least MIN_SONGS_FOR_DEFAULT_GENRE of
    its songs share that family, so one song can't steer a whole mood
    (e.g. the only "chill" song is dancehall, which shouldn't make chill = latin).
    """
    counts: Dict[str, Counter] = {}
    for s in songs:
        if "mood_family" not in s:
            normalize_song(s)
        counts.setdefault(s["mood_family"], Counter())[s["genre_family"]] += 1
    defaults = {}
    for mood, c in counts.items():
        genre, n = c.most_common(1)[0]
        if n >= MIN_SONGS_FOR_DEFAULT_GENRE:
            defaults[mood] = genre
    return defaults


def _read_catalog() -> List[Dict]:
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        return [normalize_song({"genre": r["genre"], "mood": r["mood"]}) for r in csv.DictReader(f)]


class KeywordMoodModel:
    """
    Counts keyword hits per emotion (confidence = the winner's share of hits),
    then sets the profile's genre:
    1. a genre the user names, e.g. "some blues" -> blues
    2. else, if mood words were found, the catalog's usual genre family for
       that mood (e.g. angry -> rock)
    3. else no genre preference
    """

    name = "keyword"

    def __init__(self, songs: Optional[List[Dict]] = None):
        songs = songs if songs is not None else _read_catalog()
        self.genres = sorted({s["genre"].lower() for s in songs})
        self.mood_genre = genre_for_each_mood(songs)

    def predict(self, message: str) -> MoodPrediction:
        text = message.lower()
        hits = {
            emotion: sum(1 for kw in words if _contains(text, kw))
            for emotion, words in KEYWORDS.items()
        }
        best = max(hits, key=hits.get)
        total = sum(hits.values())
        emotion = best if total else "neutral"
        profile = profile_for_emotion(emotion)

        named = find_named_genre(text, self.genres)
        if named:
            profile["favorite_genre"] = named
        elif total:
            profile["favorite_genre"] = self.mood_genre.get(profile["favorite_mood"], "")

        confidence = hits[best] / total if total else 0.0
        return MoodPrediction(emotion, confidence, profile, self.name)


# ---------------------------------------------------------------------------
# Fine-tuned classifier
# ---------------------------------------------------------------------------

class FineTunedMoodModel:
    """Loads the DistilBERT classifier saved by training/train_mood.py."""

    name = "fine-tuned"

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        # Imported here so the rest of the app works without torch installed.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, message: str) -> MoodPrediction:
        inputs = self.tokenizer(message, return_tensors="pt", truncation=True, max_length=128)
        with self._torch.no_grad():
            probs = self.model(**inputs).logits.softmax(dim=-1)[0]
        idx = int(probs.argmax())
        emotion = self.model.config.id2label[idx]
        return MoodPrediction(emotion, float(probs[idx]), profile_for_emotion(emotion), self.name)


def load_mood_model(model_dir: Path = DEFAULT_MODEL_DIR, songs: Optional[List[Dict]] = None):
    """Returns the fine-tuned model if it's trained and loadable, else the keyword baseline."""
    if Path(model_dir).joinpath("config.json").exists():
        try:
            return FineTunedMoodModel(model_dir)
        except ImportError:
            print("Fine-tuned model found but torch/transformers aren't installed; "
                  "using the keyword baseline. See requirements-train.txt.")
    return KeywordMoodModel(songs)
