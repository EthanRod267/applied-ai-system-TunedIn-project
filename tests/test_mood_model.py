from src.mood_model import (EMOTION_PROFILES, KeywordMoodModel, genre_for_each_mood,
                            load_mood_model, profile_for_emotion)
from src.recommender import load_songs


def test_every_emotion_maps_to_a_mood_with_songs():
    catalog_moods = {s["mood_family"] for s in load_songs("data/songs.csv")}
    for emotion, (mood, energy, valence, _) in EMOTION_PROFILES.items():
        assert mood in catalog_moods, f"{emotion} -> {mood} has no songs in songs.csv"
        assert 0 <= energy <= 1 and 0 <= valence <= 1


def test_keyword_model_reads_obvious_moods():
    model = KeywordMoodModel()
    assert model.predict("I'm so angry right now").profile["favorite_mood"] == "angry"
    assert model.predict("feeling really sad and lonely").profile["favorite_mood"] == "sad"
    assert model.predict("so excited for the party!!!").profile["favorite_mood"] == "euphoric"


def test_keyword_model_matches_whole_words_only():
    # "mad" must not fire inside "made"
    assert KeywordMoodModel().predict("I made pasta").emotion == "neutral"


def test_no_keywords_means_zero_confidence():
    pred = KeywordMoodModel().predict("asdfghjkl")
    assert pred.confidence == 0.0
    assert pred.profile == profile_for_emotion("neutral")


def test_loader_falls_back_to_keyword_without_trained_model(tmp_path):
    assert isinstance(load_mood_model(tmp_path), KeywordMoodModel)


def test_keyword_model_picks_the_usual_genre_family_for_the_mood():
    songs = load_songs("data/songs.csv")
    profile = KeywordMoodModel(songs).predict("date night with my partner").profile
    assert profile["favorite_mood"] == "romantic"
    assert profile["favorite_genre"] == genre_for_each_mood(songs).get("romantic", "")


def test_single_song_moods_get_no_default_genre():
    songs = [{"genre": "dancehall", "mood": "chill"}, {"genre": "pop", "mood": "upbeat"},
             {"genre": "pop", "mood": "sunny"}]
    defaults = genre_for_each_mood(songs)
    assert "chill" not in defaults      # only one chill song
    assert defaults["happy"] == "pop"   # two pop songs share the happy family


def test_mood_default_genre_counts_families():
    songs = [
        {"genre": "alternative rock", "mood": "jealous"},
        {"genre": "indie rock", "mood": "frustrated"},
        {"genre": "pop", "mood": "bitter"},
    ]
    assert genre_for_each_mood(songs)["angry"] == "rock"  # 2 rock-family songs beat 1 pop


def test_named_genre_beats_the_mood_default():
    pred = KeywordMoodModel().predict("I'm sad, play me some indie rock")
    assert pred.profile["favorite_mood"] == "sad"
    assert pred.profile["favorite_genre"] == "indie rock"


def test_named_genre_works_without_mood_words():
    assert KeywordMoodModel().predict("pump me up with some edm").profile["favorite_genre"] == "electronic"
    assert KeywordMoodModel().predict("put on some rap").profile["favorite_genre"] == "hip-hop"
