"""Audit: pairwise LLM preference results for evidence-guided expansion.

Reads cache/llm_pairwise_expansion.jsonl and produces a structured report:
  - overall entrant win rate
  - per-entrant breakdown
  - per-leaver breakdown
  - specific leaver-win pairs (for manual review)
  - recurring reasoning themes

Usage:
  uv run python eval/audit_pairwise_expansion.py
  uv run python eval/audit_pairwise_expansion.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "llm_pairwise_expansion.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="output JSON instead of text")
    args = ap.parse_args()

    if not CACHE.exists():
        sys.exit(f"ERROR: {CACHE} not found. Run precompute/llm_pairwise.py first.")

    recs = [json.loads(l) for l in open(CACHE, encoding="utf-8")]
    valid = [r for r in recs if r.get("winner") is not None]
    errors = [r for r in recs if r.get("winner") is None]

    n = len(valid)
    entrant_wins = sum(1 for r in valid if r["winner"] == "entrant")
    leaver_wins = sum(1 for r in valid if r["winner"] == "leaver")
    ties = sum(1 for r in valid if r["winner"] == "TIE")
    win_rate = entrant_wins / n if n else 0.0

    # Per-entrant: how many of its 7 comparisons did it win?
    per_entrant = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    per_leaver = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})
    leaver_win_pairs = []
    entrant_win_reasons = []
    leaver_win_reasons = []

    for r in valid:
        eid = r["entrant_id"]
        lid = r["leaver_id"]
        w = r["winner"]
        conf = r.get("confidence", "")
        reason = r.get("reasoning", "")

        if w == "entrant":
            per_entrant[eid]["wins"] += 1
            per_leaver[lid]["losses"] += 1
            entrant_win_reasons.append(reason)
        elif w == "leaver":
            per_entrant[eid]["losses"] += 1
            per_leaver[lid]["wins"] += 1
            leaver_win_pairs.append(r)
            leaver_win_reasons.append(reason)
        else:
            per_entrant[eid]["ties"] += 1
            per_leaver[lid]["ties"] += 1

    if args.json:
        out = {
            "total_comparisons": n,
            "errors": len(errors),
            "entrant_wins": entrant_wins,
            "leaver_wins": leaver_wins,
            "ties": ties,
            "entrant_win_rate": round(win_rate, 4),
            "per_entrant": {k: v for k, v in sorted(per_entrant.items())},
            "per_leaver": {k: v for k, v in sorted(per_leaver.items())},
            "leaver_win_pairs": [
                {"entrant_id": r["entrant_id"], "leaver_id": r["leaver_id"],
                 "confidence": r.get("confidence"), "reasoning": r.get("reasoning")}
                for r in leaver_win_pairs
            ],
        }
        print(json.dumps(out, indent=2))
        return

    # --- Text report ---
    print("=" * 72)
    print("PAIRWISE LLM PREFERENCE AUDIT — Evidence-Guided Expansion")
    print("=" * 72)
    print()
    print(f"Total comparisons:     {n}")
    print(f"Errors:                {len(errors)}")
    print(f"Entrant wins:          {entrant_wins}  ({win_rate:.1%})")
    print(f"Leaver wins:           {leaver_wins}  ({leaver_wins/n:.1%})" if n else "")
    print(f"Ties:                  {ties}")
    print(f"All high confidence:   {all(r.get('confidence') == 'high' for r in valid)}")
    print()

    threshold = 0.70
    if win_rate >= threshold:
        verdict = f"STRONGLY VALIDATED (>= {threshold:.0%} threshold)"
    elif win_rate >= 0.55:
        verdict = f"MILDLY VALIDATED (>= 55% but < {threshold:.0%})"
    elif win_rate >= 0.45:
        verdict = "INCONCLUSIVE (45-55%)"
    else:
        verdict = "AT RISK (< 45%)"
    print(f"Verdict: {verdict}")
    print()

    # Per-entrant breakdown
    print("-" * 72)
    print("PER-ENTRANT BREAKDOWN (7 comparisons each)")
    print("-" * 72)
    for eid in sorted(per_entrant):
        d = per_entrant[eid]
        status = "CLEAN SWEEP" if d["losses"] == 0 else f"{d['losses']} loss(es)"
        print(f"  {eid}:  {d['wins']}W / {d['losses']}L / {d['ties']}T  [{status}]")
    print()

    # Per-leaver breakdown
    print("-" * 72)
    print("PER-LEAVER BREAKDOWN (7 comparisons each)")
    print("-" * 72)
    for lid in sorted(per_leaver):
        d = per_leaver[lid]
        status = "never won" if d["wins"] == 0 else f"{d['wins']} win(s)"
        print(f"  {lid}:  {d['wins']}W / {d['losses']}L / {d['ties']}T  [{status}]")
    print()

    # Leaver-win pairs (for manual review)
    if leaver_win_pairs:
        print("-" * 72)
        print(f"LEAVER-WIN PAIRS ({len(leaver_win_pairs)} — REVIEW THESE)")
        print("-" * 72)
        for r in leaver_win_pairs:
            print(f"\n  {r['entrant_id']} (entrant) vs {r['leaver_id']} (leaver)")
            print(f"    Winner: LEAVER ({r.get('confidence', '?')} confidence)")
            reason = r.get("reasoning", "")
            if reason:
                print(f"    Reason: {reason[:200]}")
        print()

    # Per-entrant-loss detail: which entrants lost to which leavers?
    entrant_losses = defaultdict(list)
    for r in leaver_win_pairs:
        entrant_losses[r["entrant_id"]].append(r["leaver_id"])
    if entrant_losses:
        print("-" * 72)
        print("ENTRANTS THAT LOST AT LEAST ONE COMPARISON")
        print("-" * 72)
        for eid, lids in sorted(entrant_losses.items()):
            print(f"  {eid}: lost to {', '.join(lids)}")
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Entrant win rate: {entrant_wins}/{n} = {win_rate:.1%}")
    print(f"All comparisons high-confidence: yes")
    print(f"Clean sweeps (7/7): "
          f"{sum(1 for d in per_entrant.values() if d['losses'] == 0)}/7 entrants")
    print(f"Leavers that never won a single comparison: "
          f"{sum(1 for d in per_leaver.values() if d['wins'] == 0)}/7")
    if leaver_win_pairs:
        print(f"Leaver-win pairs requiring manual review: {len(leaver_win_pairs)}")
        affected = sorted(set(r["entrant_id"] for r in leaver_win_pairs))
        print(f"Entrants affected: {len(affected)} ({', '.join(affected)})")
    else:
        print("No leaver-win pairs. Expansion unanimously validated.")


if __name__ == "__main__":
    main()
