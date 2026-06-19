"""Audit: role-aware expert mixture — hidden gems and dimension dominance.

Computes 7-dimensional expert scores for all ranked candidates, then answers:
1. Are there hidden gems? (ranked >100 overall but top-50 on a key dimension)
2. Does the top-100 dominate on all dimensions?
3. What does the dimension profile look like for borderline candidates?

Usage:
  uv run python eval/audit_expert_mixture.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.llm_expansion import load_expansion_grades, apply_expansion  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402
from src.expert_scores import expert_scores, expert_profile_str, DIMENSIONS  # noqa: E402

TOP_N = 100
HIDDEN_GEM_RANK = 500  # candidates ranked up to here are candidates for hidden gem check
DIM_TOP = 50  # top-N per dimension to check for hidden gems
PRINT_GEMS = 20


def apply_final_stages(scored):
    """Match rank.py's default order: cached LLM rerank + expansion."""
    grades = load_grades()
    if grades:
        scored = apply_rerank(scored, window=200, grades=grades)

    exp = load_expansion_grades()
    if exp:
        ev_map = {r["candidate_id"]: evidence_graph_score(r["candidate"])["score"]
                  for r in scored}
        base_full = {}
        for line in open(ROOT / "cache" / "llm_rerank.jsonl", encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("fit_score") is not None:
                base_full[d["candidate_id"]] = d
        scored, _audit = apply_expansion(scored, base_full, exp, lambda cid: ev_map.get(cid, 0.0))
    return scored


def main():
    print("Loading candidates and ranking ...")
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    ranked, ref = score_all(candidates)
    ranked = apply_final_stages(ranked)
    print(f"  {len(ranked)} candidates scored, ref_date={ref}")

    # Compute expert scores for all ranked candidates
    print("Computing expert scores ...")
    for i, r in enumerate(ranked):
        r["expert"] = expert_scores(r["candidate"], ref)
    print(f"  Done ({len(ranked)} profiles)")

    overall_ids = [r["candidate_id"] for r in ranked]
    top100_ids = set(overall_ids[:TOP_N])

    # --- Per-dimension rankings ---
    print()
    print("=" * 80)
    print("PER-DIMENSION TOP-10")
    print("=" * 80)
    for dim in DIMENSIONS:
        dim_ranked = sorted(ranked, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        in_top100 = sum(1 for r in dim_ranked[:DIM_TOP] if r["candidate_id"] in top100_ids)
        print(f"\n  {dim.upper()}  (top-{DIM_TOP} overlap with overall top-100: {in_top100}/{DIM_TOP})")
        for j, r in enumerate(dim_ranked[:10]):
            rank = overall_ids.index(r["candidate_id"]) + 1
            mark = "  " if rank <= TOP_N else "★ "  # star = hidden gem in top-10
            val = r["expert"][dim]
            print(f"    {mark}#{rank:>5}  {r['candidate_id']}  {val:.3f}")

    # --- Hidden gems: ranked >100 but top-50 on a dimension ---
    print()
    print("=" * 80)
    print(f"HIDDEN GEMS (ranked {TOP_N + 1}-{HIDDEN_GEM_RANK} and top-{DIM_TOP} on a dimension)")
    print("=" * 80)

    hidden_gems = {}  # candidate_id → {dims: [...], scores: {...}}
    for dim in DIMENSIONS:
        dim_ranked = sorted(ranked, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        for r in dim_ranked[:DIM_TOP]:
            rank = overall_ids.index(r["candidate_id"]) + 1
            if TOP_N < rank <= HIDDEN_GEM_RANK:
                cid = r["candidate_id"]
                if cid not in hidden_gems:
                    hidden_gems[cid] = {"overall_rank": rank, "dims": [], "expert": r["expert"]}
                hidden_gems[cid]["dims"].append((dim, r["expert"][dim]))

    if hidden_gems:
        # Sort by number of dimensions they're strong in, then by overall rank
        sorted_gems = sorted(hidden_gems.items(),
                             key=lambda x: (-len(x[1]["dims"]), x[1]["overall_rank"]))
        print(f"\n  {len(hidden_gems)} hidden gem(s) found:\n")
        for cid, info in sorted_gems[:PRINT_GEMS]:
            print(f"  #{info['overall_rank']:>5}  {cid}")
            print(f"          Strong on {len(info['dims'])} dimension(s):")
            for dim, val in info["dims"]:
                print(f"            {dim}: {val:.3f}")
            print(f"          Profile: {expert_profile_str(info['expert'])}")
            print()
        if len(sorted_gems) > PRINT_GEMS:
            print(f"  ... {len(sorted_gems) - PRINT_GEMS} more omitted from console output")
            print()
    else:
        print("\n  No hidden gems found. Top-100 dominates all dimensions.")

    # --- Top-100 dimension dominance ---
    print("=" * 80)
    print("TOP-100 DIMENSION DOMINANCE")
    print("=" * 80)
    print(f"\n  For each dimension: how many of the top-{DIM_TOP} are in the overall top-100?")
    print(f"  (High overlap = top-100 dominates that dimension)")
    print()
    for dim in DIMENSIONS:
        dim_ranked = sorted(ranked, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        overlap = sum(1 for r in dim_ranked[:DIM_TOP] if r["candidate_id"] in top100_ids)
        bar = "#" * overlap + "-" * (DIM_TOP - overlap)
        pct = overlap / DIM_TOP * 100
        print(f"  {dim:<20} {overlap:>2}/{DIM_TOP}  {bar}  ({pct:.0f}%)")

    # --- Mean expert scores: top-100 vs next-100 vs hidden gems ---
    print()
    print("=" * 80)
    print("MEAN EXPERT SCORES BY GROUP")
    print("=" * 80)
    groups = {
        "top-100": ranked[:TOP_N],
        "ranks 101-200": ranked[TOP_N:200],
        "ranks 201-500": ranked[200:500],
    }
    if hidden_gems:
        gem_ids = set(hidden_gems.keys())
        groups["hidden gems"] = [r for r in ranked if r["candidate_id"] in gem_ids]

    print()
    header = f"  {'Group':<16}" + "".join(f"{d[:8]:>10}" for d in DIMENSIONS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, group in groups.items():
        if not group:
            continue
        means = {d: sum(r["expert"][d] for r in group) / len(group) for d in DIMENSIONS}
        row = f"  {name:<16}" + "".join(f"{means[d]:>10.3f}" for d in DIMENSIONS)
        print(row)

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_gems = len(hidden_gems)
    multi_dim_gems = sum(1 for g in hidden_gems.values() if len(g["dims"]) >= 3)
    print(f"  Hidden gems (ranked {TOP_N + 1}-{HIDDEN_GEM_RANK}, top-{DIM_TOP} on a dimension): {total_gems}")
    print(f"  Multi-dimension gems (strong on 3+ dims): {multi_dim_gems}")
    dominance = {}
    for dim in DIMENSIONS:
        dim_ranked = sorted(ranked, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        dominance[dim] = sum(1 for r in dim_ranked[:DIM_TOP] if r["candidate_id"] in top100_ids)
    weakest_dim = min(DIMENSIONS, key=lambda d: dominance[d])
    print(f"  Lowest top-100 dimension dominance: {weakest_dim}")


if __name__ == "__main__":
    main()
