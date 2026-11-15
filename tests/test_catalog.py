"""Catalog normalization: every genre and mood in songs.csv must be mapped."""

from src.catalog import (GENRE_FAMILIES, MOOD_FAMILIES, SUBGENRE_TO_FAMILY, contains_phrase,
                         find_named_genre, genre_family, mood_family, norm)
from src.recommender import load_songs

SONGS = load_songs("data/songs.csv")


def test_every_catalog_genre_has_a_family():
    missing = sorted({s["genre"] for s in SONGS if s["genre"].lower() not in SUBGENRE_TO_FAMILY})
    assert not missing, f"add these genres to GENRE_FAMILIES in src/catalog.py: {missing}"


def test_every_catalog_mood_has_a_family():
    missing = sorted({s["mood"] for s in SONGS if s["mood"].lower() not in MOOD_FAMILIES})
    assert not missing, f"add these moods to MOOD_FAMILIES in src/catalog.py: {missing}"


def test_families_map_to_themselves():
    for family in GENRE_FAMILIES:
        assert genre_family(family) == family
    for canonical in set(MOOD_FAMILIES.values()):
        assert mood_family(canonical) == canonical


def test_family_lookup_is_case_insensitive_and_handles_aliases():
    assert genre_family("Alternative Rock") == "rock"
    assert genre_family("rap") == "hip-hop"
    assert genre_family("k-pop") == "k-pop"  # unknown genres are their own family
    assert mood_family("Heartbroken") == "sad"


def test_norm_strips_accents_and_curly_quotes():
    assert norm("Señorita") == "senorita"
    assert norm("Can’t  Hold Us") == "can't hold us"


def test_contains_phrase_is_whole_word():
    assert contains_phrase("some Rock please", "rock")
    assert not contains_phrase("I like Rockstar", "rock")
    assert contains_phrase("listening to Mr. Brightside", "Mr. Brightside")


def test_find_named_genre_prefers_the_most_specific_name():
    genres = [s["genre"] for s in SONGS]
    assert find_named_genre("some pop rock please", genres) == "pop rock"
    assert find_named_genre("some rock please", genres) == "rock"
    assert find_named_genre("hip-hop to hype me up", genres) == "hip-hop"
    assert find_named_genre("I made popcorn", genres) == ""
