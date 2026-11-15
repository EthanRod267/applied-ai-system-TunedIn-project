import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from src.catalog import genre_family, mood_family, norm, normalize_song
except ModuleNotFoundError:
    from catalog import genre_family, mood_family, norm, normalize_song

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: Optional[float] = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the k best-scoring songs for this user, highest first."""
        prefs = asdict(user)
        return sorted(
            self.songs,
            key=lambda song: score_song(prefs, asdict(song))[0],
            reverse=True,
        )[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns the scoring reasons for one song as a single readable string."""
        score, reasons = score_song(asdict(user), asdict(song))
        return f"Score {score:.2f}: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads the songs CSV into a list of dicts, converting numeric columns to int/float."""
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # csv gives every value as a string; convert the numeric columns
            row["id"] = int(row["id"])
            for key in ("energy", "tempo_bpm", "valence", "danceability", "acousticness"):
                row[key] = float(row[key])
            songs.append(normalize_song(row))  # adds genre_family / mood_family
    return songs

# Points per component. Mood/genre are now grouped into families (src/catalog.py),
# so a match means something; energy no longer outweighs both of them together.
WEIGHTS = {
    "genre_family": 1.0,   # same family ("rock" vs "alternative rock")
    "genre_exact": 0.5,    # bonus for the exact subgenre
    "mood": 1.5,           # same mood family ("sad" vs "heartbroken")
    "energy": 3.0,         # x (1 - |energy - target|)^2
    "valence": 2.5,        # x (1 - |valence - target|)^2, only if target_valence is set
    "acoustic": 0.5,       # x acousticness (or 1 - acousticness)
}
MAX_SCORE = sum(WEIGHTS.values())  # 9.0 with valence; profiles without it top out at 6.5


def score_song(user_prefs: Dict, song: Dict, weights: Dict = None) -> Tuple[float, List[str]]:
    """Scores one song against the user's genre, mood, energy, valence and acoustic taste."""
    w = weights or WEIGHTS
    score = 0.0
    reasons = []

    # Genre: full credit for the exact subgenre, partial for the same family.
    # Matching is case-insensitive ("Pop" == "pop").
    wanted_genre = norm(user_prefs.get("favorite_genre"))
    if wanted_genre:
        song_genre = norm(song["genre"])
        song_family = song.get("genre_family") or genre_family(song_genre)
        if wanted_genre == song_genre:
            points = w["genre_family"] + w["genre_exact"]
            score += points
            reasons.append(f"Genre matches favorite (+{points:.2f})")
        elif genre_family(wanted_genre) == song_family:
            score += w["genre_family"]
            reasons.append(f"Same genre family: {song_family} (+{w['genre_family']:.2f})")

    # Mood: compared by family, so "sad" matches "heartbroken" and "despairing".
    wanted_mood = norm(user_prefs.get("favorite_mood"))
    if wanted_mood:
        song_mood = song.get("mood_family") or mood_family(song["mood"])
        if mood_family(wanted_mood) == song_mood:
            score += w["mood"]
            reasons.append(f"Mood matches favorite (+{w['mood']:.2f})")

    # Numerical features reward closeness to the target, not higher/lower values.
    # similarity = (1 - |difference|)^2; squaring makes big misses cost much more.
    target_energy = user_prefs.get("target_energy")
    if target_energy is not None:
        energy_score = w["energy"] * max(0.0, 1.0 - abs(song["energy"] - target_energy)) ** 2
        score += energy_score
        reasons.append(f"Energy closeness (+{energy_score:.2f})")

    # Valence (how positive a song sounds) is the strongest mood signal.
    target_valence = user_prefs.get("target_valence")
    if target_valence is not None:
        valence_score = w["valence"] * max(0.0, 1.0 - abs(song["valence"] - target_valence)) ** 2
        score += valence_score
        reasons.append(f"Valence closeness (+{valence_score:.2f})")

    # Acoustic fit: acoustic fans earn points for high acousticness, others for low.
    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        fit = song["acousticness"] if likes_acoustic else 1.0 - song["acousticness"]
        acoustic_score = w["acoustic"] * fit
        score += acoustic_score
        reasons.append(f"Acoustic fit (+{acoustic_score:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Dict = None) -> List[Tuple[Dict, float, str]]:
    """Scores every song and returns the top k as (song, score, explanation), highest first."""
    # TODO: Implement scoring and ranking logic
    # Expected return format: (song_dict, score, explanation)
    # Pythonic: build the scored list with a comprehension, then use sorted().
    #
    # .sort()  -> method on a list; sorts IN PLACE, mutates that list, returns None.
    #             Use it when you own the list and don't need the original order.
    # sorted() -> built-in function; accepts any iterable, returns a NEW sorted
    #             list and leaves the original untouched.
    #
    # Here `songs` is the caller's data, so sorted() avoids mutating it
    # and lets us sort a generator directly without an intermediate list.
    scored = (
        (song, score, "; ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song, weights)]
    )
    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
