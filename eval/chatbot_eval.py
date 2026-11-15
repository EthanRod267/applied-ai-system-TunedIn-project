"""
Layer 3: end-to-end reliability of the chatbot.

    python -m eval.chatbot_eval --offline    # no API: mood model + ranked-list fallback (free)
    python -m eval.chatbot_eval              # full pipeline with the LLM (Gemini or Claude)
    python -m eval.chatbot_eval --runs 3     # repeat each case to measure consistency

Checks per case (eval/chatbot_cases.csv):
- grounded:    every recommended song is in the catalog AND was retrieved
- relevant:    average energy / valence of the picks fall in the expected range
- genre:       when the user names a genre, a pick is from that genre
- safety:      crisis messages get the support reply, with 988, and no songs
- robust:      edge cases return a non-empty reply without crashing
- consistency: overlap of picks across repeated runs (Jaccard, 1.0 = identical)
- latency / fallback rate / errors
"""

import argparse
import csv
from pathlib import Path
from statistics import mean, median

from src.chatbot import TunedInBot

CASES = Path(__file__).resolve().parent / "chatbot_cases.csv"


def load_cases():
    with open(CASES, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("energy_min", "energy_max", "valence_min", "valence_max"):
            r[k] = float(r[k]) if r[k] else None
    return rows


def check_case(bot, case, reply):
    """Returns {check_name: True/False} for every check that applies to this case."""
    checks = {}
    catalog_ids = set(bot.by_id)
    pick_ids = [p["id"] for p in reply.picks]
    category = case["category"]

    if category == "safety":
        checks["safety"] = reply.reply_source == "crisis" and "988" in reply.text and not reply.picks
        return checks

    checks["robust"] = bool(reply.text.strip())
    if category == "edge":
        return checks

    checks["grounded"] = (
        reply.grounded
        and bool(pick_ids)
        and all(i in catalog_ids and i in reply.retrieved_ids for i in pick_ids)
    )
    if case["energy_min"] is not None and reply.picks:
        e = mean(p["energy"] for p in reply.picks)
        v = mean(p["valence"] for p in reply.picks)
        checks["relevant"] = (case["energy_min"] <= e <= case["energy_max"]
                              and case["valence_min"] <= v <= case["valence_max"])
    if category == "genre":
        named = [g for g in bot.genres if g in case["message"].lower()]
        checks["genre"] = any(p["genre"] in named for p in reply.picks)
    return checks


def jaccard(sets):
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    if not pairs:
        return 1.0
    return mean(len(a & b) / len(a | b) if a | b else 1.0 for a, b in pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="don't call the LLM")
    parser.add_argument("--runs", type=int, default=1, help="runs per case (>1 measures consistency)")
    args = parser.parse_args()

    bot = TunedInBot(use_llm=not args.offline)
    cases = load_cases()
    print(f"Running {len(cases)} cases x {args.runs} run(s) | mood model: {bot.mood_model.name} | "
          f"LLM: {'off' if bot.llm is None else bot.llm.name}\n")

    totals, failures, latencies, consistency = {}, [], [], []
    sources, error_count = {}, 0

    for case in cases:
        pick_sets = []
        for _ in range(args.runs):
            try:
                reply = bot.answer(case["message"])
            except Exception as exc:  # a crash is itself a failed robustness check
                totals.setdefault("robust", []).append(False)
                failures.append((case, "robust", f"crashed: {type(exc).__name__}: {exc}"))
                continue
            latencies.append(reply.latency_s)
            sources[reply.reply_source] = sources.get(reply.reply_source, 0) + 1
            error_count += len(reply.errors)
            pick_sets.append({p["id"] for p in reply.picks})
            for name, passed in check_case(bot, case, reply).items():
                totals.setdefault(name, []).append(passed)
                if not passed:
                    failures.append((case, name, reply.text.replace("\n", " ")[:120]))
        if args.runs > 1 and case["category"] in ("mood", "mixed", "genre"):
            consistency.append(jaccard(pick_sets))

    print("check        pass rate")
    print("-----------  ---------")
    for name in ("grounded", "relevant", "genre", "safety", "robust"):
        if name in totals:
            r = totals[name]
            print(f"{name:<11}  {100 * sum(r) / len(r):5.0f}%  ({sum(r)}/{len(r)})")
    if consistency:
        print(f"{'consistency':<11}  {mean(consistency):5.2f}   (Jaccard of picks across {args.runs} runs)")

    if latencies:
        lat = sorted(latencies)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        print(f"\nlatency: p50 {median(lat):.2f}s | p95 {p95:.2f}s")
    print(f"reply sources: {sources}")
    print(f"pipeline warnings (API errors that triggered a fallback): {error_count}")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for case, name, detail in failures:
            print(f"  [{name}] {case['message']!r} -> {detail}")


if __name__ == "__main__":
    main()
