from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --- valence + functional scoring -----------------------------------------

from src.recommender import WEIGHTS, MAX_SCORE, score_song, recommend_songs

W = WEIGHTS


def _song(**overrides):
    base = dict(id=1, title="T", artist="A", genre="pop", mood="happy", energy=0.5,
                tempo_bpm=100, valence=0.5, danceability=0.5, acousticness=0.5)
    return {**base, **overrides}


def test_valence_term_rewards_closeness():
    prefs = {"target_valence": 0.2}
    close, _ = score_song(prefs, _song(valence=0.2))
    far, _ = score_song(prefs, _song(valence=0.9))
    assert close == W["valence"]  # full points for an exact match
    assert far < close


def test_perfect_match_without_valence():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.5, "likes_acoustic": False}
    score, reasons = score_song(prefs, _song())
    assert score == W["genre_family"] + W["genre_exact"] + W["mood"] + W["energy"] + W["acoustic"] * 0.5
    assert not any("Valence" in r for r in reasons)


def test_scores_never_exceed_max_score():
    prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.5,
             "target_valence": 0.5, "likes_acoustic": True}
    score, _ = score_song(prefs, _song(acousticness=1.0))
    assert score == MAX_SCORE


def test_matching_is_case_insensitive():
    prefs = {"favorite_genre": "Pop", "favorite_mood": "Happy"}
    score, _ = score_song(prefs, _song())
    assert score == W["genre_family"] + W["genre_exact"] + W["mood"]


def test_genre_family_gets_partial_credit():
    prefs = {"favorite_genre": "rock"}
    family, reasons = score_song(prefs, _song(genre="alternative rock"))
    assert family == W["genre_family"]
    assert "family" in reasons[0]
    exact, _ = score_song({"favorite_genre": "alternative rock"}, _song(genre="alternative rock"))
    assert exact > family
    other, _ = score_song(prefs, _song(genre="synthpop"))
    assert other == 0


def test_mood_matches_by_family():
    score, _ = score_song({"favorite_mood": "sad"}, _song(mood="heartbroken"))
    assert score == W["mood"]


def test_sad_profile_ranks_low_valence_song_first():
    songs = [_song(id=1, valence=0.9, mood="sad"), _song(id=2, valence=0.15, mood="sad")]
    prefs = {"favorite_mood": "sad", "target_energy": 0.5, "target_valence": 0.2}
    top = recommend_songs(prefs, songs, k=1)
    assert top[0][0]["id"] == 2


def test_recommender_class_uses_scoring():
    rec = make_small_recommender()
    user = UserProfile(favorite_genre="lofi", favorite_mood="chill", target_energy=0.4, likes_acoustic=True)
    assert rec.recommend(user, k=1)[0].genre == "lofi"
    assert rec.explain_recommendation(user, rec.songs[1]).startswith("Score")
