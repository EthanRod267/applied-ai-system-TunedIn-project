"""The evaluation runner itself: runs offline, so no API calls."""

from types import SimpleNamespace

from eval.run_eval import build_suite, check_mood_case, print_summary, run
from src.chatbot import TunedInBot
from src.mood_model import KeywordMoodModel


def offline_bot():
    return TunedInBot(mood_model=KeywordMoodModel(), use_llm=False)


def test_suite_includes_both_case_files():
    suite = build_suite()
    kinds = {kind for kind, _, _ in suite}
    assert kinds == {"mood", "chatbot"}
    assert len(build_suite("mood")) + len(build_suite("chatbot")) == len(suite)


def test_every_input_produces_a_result_and_crisis_cases_pass():
    suite = build_suite("chatbot")
    results = run(offline_bot(), suite)
    assert len(results) == len(suite)
    assert not any(r["crash"] for r in results)
    safety = [r for r in results if r["category"] == "safety"]
    assert safety and all(all(r["checks"].values()) for r in safety)


def test_mood_check_compares_mood_families():
    bot = offline_bot()
    case = {"message": "x", "expected_mood": "sad", "energy_min": 0.0, "energy_max": 1.0,
            "valence_min": 0.0, "valence_max": 1.0}
    reply = SimpleNamespace(profile={"favorite_mood": "sad", "target_energy": 0.3, "target_valence": 0.2},
                            text="ok", grounded=True, picks=[{"id": 1}], retrieved_ids=[1])
    checks = check_mood_case(bot, case, reply)
    assert checks["mood"] and checks["targets"] and checks["grounded"]


def test_crash_counts_as_a_failed_case():
    class Boom:
        confidence_threshold = 0.6
        def answer(self, message):
            raise RuntimeError("boom")

    results = run(Boom(), build_suite("chatbot")[:2])
    assert all(r["crash"] and not all(r["checks"].values()) for r in results)


def test_summary_returns_the_pass_rate(capsys):
    bot = offline_bot()
    results = run(bot, build_suite("chatbot"))
    rate = print_summary(bot, results)
    assert 0.0 <= rate <= 1.0
    assert "SUMMARY:" in capsys.readouterr().out
