"""Chatbot pipeline tests. A fake LLM stands in for Gemini, so no API calls."""

from src.chatbot import UNDERSTAND_SYSTEM, TunedInBot
from src.llm import LLMError
from src.mood_model import KeywordMoodModel, MoodPrediction, profile_for_emotion


class FakeLLM:
    """Same interface as src.llm providers. `generate` builds the reply from the allowed song IDs."""

    name = "fake-llm"

    def __init__(self, understand=None, generate=None, error=None):
        self.understand, self.generate, self.error = understand, generate, error
        self.calls = []

    def generate_json(self, system, user, schema, max_tokens):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.error:
            raise self.error
        if system == UNDERSTAND_SYSTEM:
            return self.understand
        ids = schema["properties"]["song_ids"]["items"]["enum"]
        return self.generate(ids)


def good_reply(ids):
    return {"genre": "blues", "song_ids": [ids[0], ids[1]],
            "paragraph": "Sorry today's rough. These two songs sit right where you are."}


def make_bot(llm, threshold=0.6):
    return TunedInBot(mood_model=KeywordMoodModel(), llm=llm,
                      use_llm=llm is not None, confidence_threshold=threshold)


def test_happy_path_uses_llm_and_stays_grounded():
    bot = make_bot(FakeLLM(generate=good_reply))
    reply = bot.answer("feeling really sad today")
    assert reply.reply_source == "fake-llm"
    assert reply.grounded
    assert len(reply.picks) == 2
    assert all(p["id"] in reply.retrieved_ids for p in reply.picks)
    assert reply.text == "Sorry today's rough. These two songs sit right where you are."


def test_reply_is_one_paragraph_without_lists():
    reply = make_bot(FakeLLM(generate=good_reply)).answer("feeling really sad today")
    assert "\n" not in reply.text


def test_paragraph_naming_an_unpicked_catalog_song_is_rejected():
    bot = make_bot(None)

    def sneaky(ids):
        # Picks the first retrieved song, but also names a catalog song it didn't pick.
        other = next(s for s in bot.songs if str(s["id"]) not in ids and len(s["title"].split()) > 1)
        return {"genre": "pop", "song_ids": [ids[0]],
                "paragraph": f"Try this one, or maybe {other['title']} by {other['artist']}."}

    bot.llm = FakeLLM(generate=sneaky)
    reply = bot.answer("feeling really sad today")
    assert reply.grounded is False
    assert reply.reply_source == "fallback"


def test_common_word_titles_are_not_false_alarms():
    bot = make_bot(None)
    perfect = next(s for s in bot.songs if s["title"] == "Perfect")
    stay = next(s for s in bot.songs if s["title"] == "Stay")
    prose = "This is a perfect song to stay with tonight, and it's lovely and yellow-bright."
    assert bot.mentioned_titles(prose) == []
    # ...but the same word with its artist, or in quotes, does count.
    assert perfect["id"] in bot.mentioned_titles("Perfect by Ed Sheeran is gentle.")
    assert stay["id"] in bot.mentioned_titles('Try "Stay" next.')


def test_title_matching_ignores_accents_and_curly_apostrophes():
    bot = make_bot(None)
    senorita = next(s for s in bot.songs if s["title"].startswith("Se"))
    cant = next(s for s in bot.songs if s["title"] == "Can't Hold Us")
    assert senorita["id"] in bot.mentioned_titles("Senorita by Shawn Mendes")
    assert cant["id"] in bot.mentioned_titles("Can’t Hold Us is huge")


def test_song_ids_are_sent_as_string_enum_and_converted_back():
    llm = FakeLLM(generate=good_reply)
    reply = make_bot(llm).answer("feeling really sad today")
    enum = llm.calls[-1]["schema"]["properties"]["song_ids"]["items"]["enum"]
    assert all(isinstance(i, str) for i in enum)
    assert all(isinstance(p["id"], int) for p in reply.picks)


def test_invented_song_ids_are_dropped_and_flagged():
    bad = lambda ids: {"genre": "pop", "song_ids": ["999"], "paragraph": "hi"}
    reply = make_bot(FakeLLM(generate=bad)).answer("feeling really sad today")
    assert reply.grounded is False
    assert reply.reply_source == "fallback"  # nothing valid left, so ranked list instead
    assert all(p["id"] in reply.retrieved_ids for p in reply.picks)


def test_api_failure_falls_back_to_ranked_list():
    reply = make_bot(FakeLLM(error=ConnectionError("API down"))).answer("so excited for the party!!!")
    assert reply.reply_source == "fallback"
    assert len(reply.picks) == 3
    assert reply.errors


def test_blocked_or_truncated_response_is_treated_as_failure():
    reply = make_bot(FakeLLM(error=LLMError("empty response"))).answer("feeling sad")
    assert reply.reply_source == "fallback"
    assert any("empty response" in e for e in reply.errors)


def test_low_confidence_asks_llm_for_the_mood():
    understand = {"emotion": "grief", "target_energy": 0.2, "target_valence": 0.1,
                  "likes_acoustic": True, "favorite_genre": ""}
    reply = make_bot(FakeLLM(understand=understand, generate=good_reply)).answer(
        "the house feels so quiet without her")
    assert reply.mood_source == "fake-llm"
    assert reply.profile["favorite_mood"] == "melancholy"
    assert reply.profile["target_valence"] == 0.1


def test_confident_local_model_skips_llm_understand():
    llm = FakeLLM(generate=good_reply)
    make_bot(llm, threshold=0.5).answer("I'm so angry right now")
    assert all(c["system"] != UNDERSTAND_SYSTEM for c in llm.calls)


def test_crisis_message_gets_support_and_no_api_call():
    llm = FakeLLM(generate=good_reply)
    reply = make_bot(llm).answer("I don't want to be here anymore")
    assert reply.reply_source == "crisis"
    assert "988" in reply.text
    assert not reply.picks
    assert llm.calls == []


def test_empty_message():
    assert make_bot(None).answer("   ").reply_source == "empty"


def test_works_fully_offline_with_a_paragraph():
    reply = TunedInBot(mood_model=KeywordMoodModel(), use_llm=False).answer("chill sunday")
    assert reply.reply_source == "fallback"
    assert reply.picks
    assert "\n" not in reply.text
    assert all(p["title"] in reply.text for p in reply.picks)


def test_named_genre_is_detected_offline():
    bot = TunedInBot(mood_model=KeywordMoodModel(), use_llm=False)
    assert bot.named_genre("pump me up with some EDM") == "electronic"
    assert bot.named_genre("some indie pop please") == "indie pop"
    assert bot.named_genre("I made popcorn") == ""
    reply = bot.answer("feeling sad and I want some rock")
    assert any(p["genre_family"] == "rock" for p in reply.picks)


class AlwaysNeutral:
    """Stands in for an undertrained fine-tuned model."""
    name = "fine-tuned"

    def predict(self, message):
        return MoodPrediction("neutral", 0.06, profile_for_emotion("neutral"), self.name)


def test_weak_primary_model_falls_back_to_keywords_when_llm_fails():
    bot = TunedInBot(mood_model=AlwaysNeutral(), llm=FakeLLM(error=ConnectionError("down")))
    reply = bot.answer("I'm in a sad mood right now")
    assert reply.mood_source == "keyword"
    assert reply.profile["favorite_mood"] == "sad"
    assert reply.errors  # the failed LLM call is still reported


def test_weak_primary_model_kept_when_keywords_find_nothing():
    bot = TunedInBot(mood_model=AlwaysNeutral(), use_llm=False)
    assert bot.answer("asdfghjkl").mood_source == "fine-tuned"
