"""
Catalog normalization: groups the many specific genres and moods in
data/songs.csv into a small set of families the scorer and mood models share.

- Genres: 26 subgenres ("alternative rock", "edm pop", ...) -> 8 families.
  The CSV keeps the specific subgenre; scoring gives full credit for the exact
  subgenre and partial credit for the same family.
- Moods: 37 descriptive moods ("heartbroken", "wistful", ...) -> the canonical
  moods the mood models output (see mood_model.EMOTION_PROFILES).

When you add songs with a new genre or mood, add it here. tests/test_catalog.py
fails until every catalog genre and mood is mapped.
"""

import re
import unicodedata
from typing import Dict, Iterable

GENRE_FAMILIES: Dict[str, list] = {
    "pop":        ["pop", "synthpop", "indie pop", "alt pop", "acoustic pop", "pop ballad",
                   "piano ballad", "disco pop", "funk pop", "pop funk"],
    "rock":       ["rock", "alternative rock", "indie rock", "pop rock", "soul rock", "progressive rock"],
    "metal":      ["metal", "nu metal"],
    "hip-hop":    ["hip-hop", "hip hop", "trap"],
    "r&b":        ["r&b", "soul pop"],
    "electronic": ["electronic", "edm pop", "edm folk"],
    "folk":       ["folk", "indie folk"],
    "latin":      ["latin", "latin pop", "dancehall"],
}

# Extra words a user might type for a family ("rap", "edm", "rnb", ...).
GENRE_ALIASES: Dict[str, str] = {
    "rap": "hip-hop", "hiphop": "hip-hop",
    "edm": "electronic", "dance": "electronic", "house": "electronic", "techno": "electronic",
    "rnb": "r&b", "r and b": "r&b", "soul": "r&b",
    "acoustic": "folk",
    "reggaeton": "latin", "latino": "latin", "caribbean": "latin",
    "punk": "rock", "grunge": "rock",
}

# Catalog mood -> canonical mood. Canonical moods map to themselves.
MOOD_FAMILIES: Dict[str, str] = {
    # canonical
    "happy": "happy", "chill": "chill", "intense": "intense", "moody": "moody",
    "confident": "confident", "nostalgic": "nostalgic", "melancholy": "melancholy",
    "angry": "angry", "romantic": "romantic", "uplifting": "uplifting",
    "euphoric": "euphoric", "energetic": "energetic", "playful": "playful", "sad": "sad",
    # catalog moods
    "upbeat": "happy", "carefree": "happy", "sunny": "happy",
    "melancholic": "melancholy", "yearning": "melancholy", "longing": "melancholy",
    "regretful": "melancholy",
    "heartbroken": "sad", "despairing": "sad",
    "wistful": "nostalgic", "sentimental": "nostalgic",
    "flirty": "playful",
    "sultry": "romantic", "tender": "romantic",
    "dark": "moody", "anxious": "moody", "paranoid": "moody",
    "obsessive": "intense", "restless": "intense", "dramatic": "intense",
    "hopeful": "uplifting",
    "triumphant": "euphoric",
    "epic": "energetic",
    "defiant": "confident", "empowered": "confident", "cocky": "confident",
    "jealous": "angry", "frustrated": "angry", "bitter": "angry",
}

SUBGENRE_TO_FAMILY = {sub: fam for fam, subs in GENRE_FAMILIES.items() for sub in subs}


def norm(text) -> str:
    """Lowercase, strip accents, straighten curly quotes, collapse spaces."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def genre_family(genre: str) -> str:
    """Family for a subgenre, family name, or alias. Unknown genres are their own family."""
    g = norm(genre)
    if g in GENRE_FAMILIES:
        return g
    return SUBGENRE_TO_FAMILY.get(g) or GENRE_ALIASES.get(g) or g


def mood_family(mood: str) -> str:
    """Canonical mood for a catalog mood. Unknown moods are their own family."""
    m = norm(mood)
    return MOOD_FAMILIES.get(m, m)


def normalize_song(song: Dict) -> Dict:
    """Adds genre_family and mood_family to a song dict (in place) and returns it."""
    song["genre_family"] = genre_family(song["genre"])
    song["mood_family"] = mood_family(song["mood"])
    return song


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word, normalized match: "rock" is found in "some Rock please", not in "rockstar"."""
    p = norm(phrase)
    if not p:
        return False
    return re.search(rf"(?<![\w']){re.escape(p)}(?![\w'])", norm(text)) is not None


def find_named_genre(text: str, catalog_genres: Iterable[str]) -> str:
    """
    Returns the genre a user names, most specific first:
    1. a catalog subgenre ("indie pop"), longest match first
    2. a family name or alias ("rock", "rap" -> hip-hop, "edm" -> electronic)
    Returns "" if none is named.
    """
    for genre in sorted({norm(g) for g in catalog_genres}, key=len, reverse=True):
        if contains_phrase(text, genre):
            return genre
    names = list(GENRE_FAMILIES) + list(GENRE_ALIASES)
    for name in sorted(names, key=len, reverse=True):
        if contains_phrase(text, name):
            return genre_family(name)
    return ""
