"""Chatbot pipeline tests. A fake client stands in for Claude, so no API calls."""

import json
from types import SimpleNamespace

from src.chatbot import UNDERSTAND_SYSTEM, TunedInBot
from src.mood_model import KeywordMoodModel


class FakeClient:
    """Mimics client.beta.messages.create. `generate` builds the reply from the retrieved IDs."""

    def __init__(self, understand=None, generate=None, fail=False, stop_reason="end_turn"):
        self.understand, self.generate = understand, generate
        self.fail, self.stop_reason = fail, stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("API down")
        if kwargs["system"] == UNDERSTAND_SYSTEM:
            data = self.understand
        else:
            ids = kwargs["output_config"]["format"]["schema"]["properties"]["picks"]["items"]["properties"]["song_id"]["enum"]
            data = self.generate(ids)
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(data))])


def good_reply(ids):
    return {"intro": "Sorry today's rough.", "genre": "blues",
            "picks": [{"song_id": ids[0], "why": "fits"}, {"song_id": ids[1], "why": "also fits"}]}


def make_bot(client, threshold=0.6):
    return TunedInBot(mood_model=KeywordMoodModel(), client=client,
                      use_claude=client is not None, confidence_threshold=threshold)


def test_happy_path_uses_claude_and_stays_grounded():
    bot = make_bot(FakeClient(generate=good_reply))
    reply = bot.answer("feeling really sad today")
    assert reply.reply_source == "claude"
    assert reply.grounded
    assert len(reply.picks) == 2
    assert all(p["id"] in reply.retrieved_ids for p in reply.picks)
    assert reply.picks[0]["title"] in reply.text


def test_invented_song_ids_are_dropped_and_flagged():
    bad = lambda ids: {"intro": "hi", "genre": "pop", "picks": [{"song_id": 999, "why": "made up"}]}
    reply = make_bot(FakeClient(generate=bad)).answer("feeling really sad today")
    assert reply.grounded is False
    assert reply.reply_source == "fallback"  # nothing valid left, so ranked list instead
    assert all(p["id"] in reply.retrieved_ids for p in reply.picks)


def test_api_failure_falls_back_to_ranked_list():
    reply = make_bot(FakeClient(fail=True)).answer("so excited for the party!!!")
    assert reply.reply_source == "fallback"
    assert len(reply.picks) == 3
    assert reply.errors


def test_refusal_is_treated_as_failure():
    reply = make_bot(FakeClient(generate=good_reply, stop_reason="refusal")).answer("feeling sad")
    assert reply.reply_source == "fallback"
    assert any("refusal" in e for e in reply.errors)


def test_low_confidence_asks_claude_for_the_mood():
    understand = {"emotion": "grief", "target_energy": 0.2, "target_valence": 0.1,
                  "likes_acoustic": True, "favorite_genre": ""}
    client = FakeClient(understand=understand, generate=good_reply)
    reply = make_bot(client).answer("the house feels so quiet without her")
    assert reply.mood_source == "claude"
    assert reply.profile["favorite_mood"] == "melancholy"
    assert reply.profile["target_valence"] == 0.1


def test_confident_local_model_skips_claude_understand():
    client = FakeClient(generate=good_reply)
    make_bot(client, threshold=0.5).answer("I'm so angry right now")
    assert all(c["system"] != UNDERSTAND_SYSTEM for c in client.calls)


def test_crisis_message_gets_support_and_no_api_call():
    client = FakeClient(generate=good_reply)
    reply = make_bot(client).answer("I don't want to be here anymore")
    assert reply.reply_source == "crisis"
    assert "988" in reply.text
    assert not reply.picks
    assert client.calls == []


def test_empty_message():
    assert make_bot(None).answer("   ").reply_source == "empty"


def test_works_fully_offline():
    reply = TunedInBot(mood_model=KeywordMoodModel(), use_claude=False).answer("chill sunday")
    assert reply.reply_source == "fallback"
    assert reply.picks


def test_named_genre_is_detected_offline():
    bot = TunedInBot(mood_model=KeywordMoodModel(), use_claude=False)
    assert bot.named_genre("pump me up with some EDM") == "edm"
    assert bot.named_genre("some indie pop please") == "indie pop"
    assert bot.named_genre("I made popcorn") == ""
    reply = bot.answer("feeling sad and I want some blues")
    assert any(p["genre"] == "blues" for p in reply.picks)
