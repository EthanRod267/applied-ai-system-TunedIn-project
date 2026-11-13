from src.mood_model import EMOTION_PROFILES, KeywordMoodModel, load_mood_model, profile_for_emotion
from src.recommender import load_songs


def test_every_emotion_maps_to_a_catalog_mood():
    catalog_moods = {s["mood"] for s in load_songs("data/songs.csv")}
    for emotion, (mood, energy, valence, _) in EMOTION_PROFILES.items():
        assert mood in catalog_moods, f"{emotion} -> {mood} is not in songs.csv"
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
