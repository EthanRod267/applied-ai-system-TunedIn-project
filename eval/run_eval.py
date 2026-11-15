"""
Runs TunedIn on every predefined input and prints a pass/fail report.

    python -m eval.run_eval --offline          # free: local mood model + offline replies
    python -m eval.run_eval                    # full system with Gemini (uses the API)
    python -m eval.run_eval --offline -v       # also print the reply for each failure
    python -m eval.run_eval --min-pass 0.8     # exit code 1 if under 80% of cases pass

Inputs (both files are predefined test sets):
- eval/mood_cases.csv     30 messages with the expected mood and energy/valence targets
- eval/chatbot_cases.csv  messages covering moods, mixed feelings, named genres,
                          crisis messages and edge cases

A case passes only if every check that applies to it passes:
- mood:      the mood was read as the expected mood family
- targets:   target energy and valence fall in the expected range
- grounded:  every recommended song came from the catalog and the retrieved set
- relevant:  the picks' average energy/valence fit the mood
- genre:     a named genre ("some rock") is honored
- safety:    crisis messages get the 988 support reply and no songs
- robust:    edge cases (empty, emoji, gibberish) still get a reply
"""

import argparse
import sys
from statistics import mean, median

from eval.chatbot_eval import check_case
from eval.chatbot_eval import load_cases as load_chatbot_cases
from eval.mood_eval import load_cases as load_mood_cases
from src.catalog import mood_family
from src.chatbot import TunedInBot

CHECK_ORDER = ["mood", "targets", "grounded", "relevant", "genre", "safety", "robust"]


def check_mood_case(bot, case, reply):
    checks = check_case(bot, {**case, "category": "mood", "energy_min": None}, reply)
    if reply.profile:
        p = reply.profile
        checks["mood"] = mood_family(p["favorite_mood"]) == mood_family(case["expected_mood"])
        checks["targets"] = (case["energy_min"] <= p["target_energy"] <= case["energy_max"]
                             and case["valence_min"] <= p["target_valence"] <= case["valence_max"])
    return checks


def build_suite(only=None):
    """Every predefined input as (suite, category, case)."""
    suite = []
    if only in (None, "mood"):
        # "reading" = mood-reading cases, so they don't merge with the chatbot file's "mood" rows
        suite += [("mood", "reading", c) for c in load_mood_cases()]
    if only in (None, "chatbot"):
        suite += [("chatbot", c["category"], c) for c in load_chatbot_cases()]
    return suite


def run(bot, suite):
    results = []
    for kind, category, case in suite:
        try:
            reply = bot.answer(case["message"])
        except Exception as exc:  # a crash fails the case instead of stopping the run
            results.append({"kind": kind, "category": category, "case": case, "reply": None,
                            "checks": {"robust": False}, "crash": f"{type(exc).__name__}: {exc}"})
            continue
        checks = check_mood_case(bot, case, reply) if kind == "mood" else check_case(bot, case, reply)
        results.append({"kind": kind, "category": category, "case": case, "reply": reply,
                        "checks": checks, "crash": None})
    return results


def describe_failure(r):
    if r["crash"]:
        return f"crashed ({r['crash']})"
    reply, case = r["reply"], r["case"]
    notes = []
    for name in CHECK_ORDER:
        if r["checks"].get(name) is False:
            if name == "mood" and reply.profile:
                notes.append(f"mood: got {reply.profile['favorite_mood']}, want {case['expected_mood']}")
            elif name == "targets" and reply.profile:
                notes.append(f"targets: energy {reply.profile['target_energy']:.2f}, "
                             f"valence {reply.profile['target_valence']:.2f}")
            else:
                notes.append(name)
    return "; ".join(notes)


def mood_label(reply):
    if reply is None or reply.mood_source is None:
        return f"- ({reply.reply_source})" if reply else "-"
    return f"{reply.profile['favorite_mood']} ({reply.mood_source}, {reply.mood_confidence:.2f})"


def print_cases(results, verbose):
    print(f"{'#':>3}  {'result':6}  {'type':7}  {'message':40}  {'mood read as (model, confidence)':34}  failed checks")
    print(f"{'-'*3}  {'-'*6}  {'-'*7}  {'-'*40}  {'-'*34}  {'-'*13}")
    for i, r in enumerate(results, start=1):
        passed = all(r["checks"].values())
        msg = r["case"]["message"] or "(empty)"
        msg = msg if len(msg) <= 40 else msg[:37] + "..."
        print(f"{i:>3}  {'PASS' if passed else 'FAIL':6}  {r['category']:7}  {msg:40}  "
              f"{mood_label(r['reply']):34}  {'' if passed else describe_failure(r)}")
        if verbose and not passed and r["reply"]:
            print(f"{'':22}reply: {r['reply'].text[:200]}")


def print_summary(bot, results):
    n = len(results)
    passed = sum(all(r["checks"].values()) for r in results)
    replies = [r["reply"] for r in results if r["reply"]]

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{n} cases passed ({100 * passed / n:.0f}%)")
    print("=" * 60)

    print("\nBy check:")
    for name in CHECK_ORDER:
        vals = [r["checks"][name] for r in results if name in r["checks"]]
        if vals:
            print(f"  {name:<9} {100 * sum(vals) / len(vals):5.0f}%  ({sum(vals)}/{len(vals)})")

    print("\nBy input type:")
    for cat in dict.fromkeys(r["category"] for r in results):
        group = [r for r in results if r["category"] == cat]
        ok = sum(all(r["checks"].values()) for r in group)
        print(f"  {cat:<9} {ok}/{len(group)}")

    # Confidence: only the local models produce a real score (Gemini always reports 1.0).
    local = [r for r in results if r["reply"] and r["reply"].mood_source in ("keyword", "fine-tuned")
             and "mood" in r["checks"]]
    print("\nConfidence (local mood model, mood-labeled cases only):")
    if local:
        threshold = bot.confidence_threshold
        avg = mean(r["reply"].mood_confidence for r in local)
        print(f"  average confidence: {avg:.2f}")
        for label, group in (
            (f"confident (>= {threshold})", [r for r in local if r["reply"].mood_confidence >= threshold]),
            (f"unsure    (<  {threshold})", [r for r in local if r["reply"].mood_confidence < threshold]),
        ):
            if group:
                acc = sum(r["checks"]["mood"] for r in group) / len(group)
                print(f"  {label}: {len(group):2d} cases, mood correct {100 * acc:.0f}%")
        print("  (if 'confident' isn't clearly more accurate than 'unsure', the threshold isn't doing its job)")
    else:
        print("  no local-model predictions (all moods were read by Gemini)")

    sources = {}
    for rep in replies:
        sources[rep.reply_source] = sources.get(rep.reply_source, 0) + 1
    warnings = sum(len(rep.errors) for rep in replies)
    lat = sorted(rep.latency_s for rep in replies)
    print("\nSystem:")
    print(f"  mood model: {bot.mood_model.name} | LLM: {bot.llm.name if bot.llm else 'off'}")
    print(f"  reply sources: {sources}")
    print(f"  API warnings that triggered a fallback: {warnings}")
    print(f"  crashes: {sum(1 for r in results if r['crash'])}")
    if lat:
        print(f"  latency: p50 {median(lat):.2f}s | max {lat[-1]:.2f}s")
    return passed / n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="don't call Gemini")
    parser.add_argument("--only", choices=["mood", "chatbot"], help="run just one case file")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the reply for each failure")
    parser.add_argument("--min-pass", type=float, default=None,
                        help="exit with code 1 if the pass rate is below this (0-1)")
    args = parser.parse_args()

    # Windows consoles can't always print emoji/accents; replace instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    bot = TunedInBot(use_llm=not args.offline)
    suite = build_suite(args.only)
    print(f"\nRunning {len(suite)} predefined inputs "
          f"({'offline' if bot.llm is None else 'with ' + bot.llm.name})\n")
    results = run(bot, suite)
    print_cases(results, args.verbose)
    rate = print_summary(bot, results)

    if args.min_pass is not None:
        verdict = "PASS" if rate >= args.min_pass else "FAIL"
        print(f"\nVerdict: {verdict} (pass rate {100 * rate:.0f}% vs required {100 * args.min_pass:.0f}%)")
        sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
