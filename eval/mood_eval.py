"""
Layer 2: how well does each model read the user's mood?

    python -m eval.mood_eval                 # keyword baseline + fine-tuned (if trained)
    python -m eval.mood_eval --with-llm      # also the LLM: Gemini or Claude (calls the API)

For each case in eval/mood_cases.csv it checks:
- mood accuracy: predicted catalog mood == expected_mood
- energy / valence in range: predicted target falls inside the expected range
- energy / valence MAE: distance from the middle of the expected range
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import mean

from src.chatbot import TunedInBot
from src.mood_model import DEFAULT_MODEL_DIR, KeywordMoodModel

CASES = Path(__file__).resolve().parent / "mood_cases.csv"


class LLMMoodModel:
    """Wraps the chatbot's LLM mood reader so it fits the predict() interface."""

    def __init__(self):
        self.bot = TunedInBot(mood_model=KeywordMoodModel())
        if self.bot.llm is None:
            raise RuntimeError("No LLM available: set GEMINI_API_KEY (or ANTHROPIC_API_KEY)")
        self.name = self.bot.llm.name

    def predict(self, message):
        return self.bot._llm_understand(message)


def load_cases():
    with open(CASES, newline="", encoding="utf-8") as f:
        return [
            {**r, **{k: float(r[k]) for k in ("energy_min", "energy_max", "valence_min", "valence_max")}}
            for r in csv.DictReader(f)
        ]


def evaluate(model, cases):
    rows = []
    for case in cases:
        start = time.perf_counter()
        try:
            pred = model.predict(case["message"])
        except Exception as exc:
            rows.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
            continue
        p = pred.profile
        rows.append({
            "case": case,
            "mood": p["favorite_mood"],
            "emotion": pred.emotion,
            "mood_ok": p["favorite_mood"] == case["expected_mood"],
            "energy_ok": case["energy_min"] <= p["target_energy"] <= case["energy_max"],
            "valence_ok": case["valence_min"] <= p["target_valence"] <= case["valence_max"],
            "energy_err": abs(p["target_energy"] - (case["energy_min"] + case["energy_max"]) / 2),
            "valence_err": abs(p["target_valence"] - (case["valence_min"] + case["valence_max"]) / 2),
            "seconds": time.perf_counter() - start,
        })
    return rows


def summarize(name, rows):
    ok = [r for r in rows if "error" not in r]
    n = len(rows)
    pct = lambda key: 100 * sum(r[key] for r in ok) / n if n else 0.0
    return {
        "model": name,
        "mood acc": f"{pct('mood_ok'):.0f}%",
        "energy in-range": f"{pct('energy_ok'):.0f}%",
        "valence in-range": f"{pct('valence_ok'):.0f}%",
        "energy MAE": f"{mean(r['energy_err'] for r in ok):.2f}" if ok else "-",
        "valence MAE": f"{mean(r['valence_err'] for r in ok):.2f}" if ok else "-",
        "errors": str(n - len(ok)),
        "avg s": f"{mean(r['seconds'] for r in ok):.2f}" if ok else "-",
    }


def print_table(dicts):
    cols = list(dicts[0])
    widths = {c: max(len(c), *(len(d[c]) for d in dicts)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for d in dicts:
        print("  ".join(d[c].ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="also evaluate the LLM (uses the API)")
    parser.add_argument("--show-misses", action="store_true", help="list every wrong mood prediction")
    args = parser.parse_args()

    cases = load_cases()
    models = [KeywordMoodModel()]
    if DEFAULT_MODEL_DIR.joinpath("config.json").exists():
        from src.mood_model import FineTunedMoodModel
        models.append(FineTunedMoodModel())
    else:
        print("(no fine-tuned model yet; run training/train_mood.py to add it to this table)\n")
    if args.with_llm:
        models.append(LLMMoodModel())

    summaries = []
    for model in models:
        rows = evaluate(model, cases)
        summaries.append(summarize(model.name, rows))
        if args.show_misses:
            print(f"\nMisses for {model.name}:")
            for r in rows:
                if "error" in r:
                    print(f"  ERROR {r['case']['message']!r}: {r['error']}")
                elif not r["mood_ok"]:
                    print(f"  {r['case']['message']!r}: got {r['mood']} ({r['emotion']}), "
                          f"expected {r['case']['expected_mood']}")

    print(f"\nMood-reading eval ({len(cases)} cases)\n")
    print_table(summaries)


if __name__ == "__main__":
    main()
