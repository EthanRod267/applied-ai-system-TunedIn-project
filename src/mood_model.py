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
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mood_classifier"

# GoEmotions label -> (song mood, target_energy, target_valence, likes_acoustic)
# Moods are the labels used in data/songs.csv. The numbers place each emotion
# on the energy (calm -> excited) and valence (sad -> happy) axes.
EMOTION_PROFILES: Dict[str, tuple] = {
    "admiration":     ("uplifting",  0.60, 0.80, False),
    "amusement":      ("playful",    0.70, 0.85, False),
    "anger":          ("angry",      0.95, 0.25, False),
    "annoyance":      ("angry",      0.80, 0.30, False),
    "approval":       ("confident",  0.65, 0.70, False),
    "caring":         ("romantic",   0.45, 0.70, True),
    "confusion":      ("moody",      0.50, 0.45, False),
    "curiosity":      ("focused",    0.45, 0.60, True),
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
    "realization":    ("focused",    0.45, 0.60, True),
    "relief":         ("relaxed",    0.35, 0.70, True),
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
    source: str           # which model produced it ("keyword", "fine-tuned", "claude")


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


class KeywordMoodModel:
    """Counts keyword hits per emotion. Confidence is the winner's share of hits."""

    name = "keyword"

    def predict(self, message: str) -> MoodPrediction:
        text = message.lower()
        hits = {
            emotion: sum(1 for kw in words if _contains(text, kw))
            for emotion, words in KEYWORDS.items()
        }
        best = max(hits, key=hits.get)
        total = sum(hits.values())
        if total == 0:
            return MoodPrediction("neutral", 0.0, profile_for_emotion("neutral"), self.name)
        return MoodPrediction(best, hits[best] / total, profile_for_emotion(best), self.name)


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


def load_mood_model(model_dir: Path = DEFAULT_MODEL_DIR):
    """Returns the fine-tuned model if it's trained and loadable, else the keyword baseline."""
    if Path(model_dir).joinpath("config.json").exists():
        try:
            return FineTunedMoodModel(model_dir)
        except ImportError:
            print("Fine-tuned model found but torch/transformers aren't installed; "
                  "using the keyword baseline. See requirements-train.txt.")
    return KeywordMoodModel()
