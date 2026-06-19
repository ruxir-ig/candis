"""Loop 6: Select candidates for Expert-Guided Expansion v2.

Selects from ranks 101-500 using expert-mixture dimensions, evidence scores,
and prior-loop watchlists. Output: a candidate ID list for LLM grading.

Selection criteria (union of all):
  1. Multi-dimension expert gem (2+ dims in per-dimension top-50)
  2. Top-25 in IR/search expert dimension
  3. Top-25 in RecSys/ranking expert dimension
  4. Top-25 in NLP/LLM expert dimension
  5. High evidence_graph (>=0.90) + high semantic (>=0.75)
  6. Prior-loop watchlist IDs

Cap: 100 candidates max. Excludes any already in base LLM rerank cache
or the original expansion cache.

Usage:
  uv run python precompute/select_expansion_v2.py --out /tmp/expansion_v2_ids.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.llm_expansion import load_expansion_grades, apply_expansion  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402
from src.expert_scores import expert_scores, DIMENSIONS  # noqa: E402

RANK_LO = 101
RANK_HI = 500
DIM_TOP = 50
CAP = 100

WATCHLIST = {
    "CAND_0065878",  # 3-dim gem, rank 127
    "CAND_0064904",  # LinkedIn, 2-dim, pairwise winner
    "CAND_0027801",  # InMobi, 2-dim
    "CAND_0094056",  # Rephrase.ai, 2-dim
    "CAND_0024620",  # PharmEasy, 2-dim
    "CAND_0018722",  # Saarthi.ai, Loop 1 pairwise winner (3/7)
}


def apply_final_stages(scored):
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
        scored, _ = apply_expansion(scored, base_full, exp, lambda cid: ev_map.get(cid, 0.0))
    return scored


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/expansion_v2_ids.txt")
    args = ap.parse_args()

    print("Loading and ranking ...")
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    ranked, ref = score_all(candidates)
    ranked = apply_final_stages(ranked)

    # Already graded candidates (exclude)
    already_graded = set()
    for cache_name in ["llm_rerank.jsonl", "llm_expansion.jsonl"]:
        p = ROOT / "cache" / cache_name
        if p.exists():
            for line in open(p):
                try:
                    d = json.loads(line)
                    if d.get("fit_score") is not None:
                        already_graded.add(d["candidate_id"])
                except json.JSONDecodeError:
                    pass
    print(f"  Already graded (exclude): {len(already_graded)}")

    # Compute expert scores for ranks 101-500
    band = ranked[RANK_LO - 1:RANK_HI]
    print(f"  Band: ranks {RANK_LO}-{RANK_HI} ({len(band)} candidates)")

    for r in band:
        r["expert"] = expert_scores(r["candidate"], ref)
        r["evidence"] = evidence_graph_score(r["candidate"])["score"]

    # Selection criteria
    selected = set()

    # 1. Multi-dimension gems (2+ dims in per-dim top-50)
    band_ids = {r["candidate_id"] for r in band}
    dim_tops = {}
    for dim in DIMENSIONS:
        dim_ranked = sorted(band, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        dim_tops[dim] = {r["candidate_id"] for r in dim_ranked[:DIM_TOP]}

    dim_counts = {}
    for cid in band_ids:
        cnt = sum(1 for dim in DIMENSIONS if cid in dim_tops[dim])
        if cnt >= 2:
            dim_counts[cid] = cnt
            selected.add(cid)
    print(f"  Multi-dim gems (2+): {len(dim_counts)}")

    # 2-4. Top-25 per key technical dimension
    for dim in ["ir_search", "recsys_ranking", "nlp_llm"]:
        dim_ranked = sorted(band, key=lambda r: (r["expert"][dim], r["candidate_id"]), reverse=True)
        for r in dim_ranked[:25]:
            selected.add(r["candidate_id"])
    print(f"  After top-25 per tech dim: {len(selected)}")

    # 5. High evidence + high semantic
    from src.semantic_matcher import semantic_scores
    sem = semantic_scores() or {}
    for r in band:
        sem_score = sem.get(r["candidate_id"], 0)
        if r["evidence"] >= 0.90 and sem_score >= 0.75:
            selected.add(r["candidate_id"])
    print(f"  After high-evidence+semantic: {len(selected)}")

    # 6. Watchlist
    for cid in WATCHLIST:
        if cid in band_ids:
            selected.add(cid)
    print(f"  After watchlist: {len(selected)}")

    # Exclude already graded
    selected = {cid for cid in selected if cid not in already_graded}
    print(f"  After excluding already-graded: {len(selected)}")

    # Cap
    selected_list = sorted(selected)
    if len(selected_list) > CAP:
        selected_list = selected_list[:CAP]
        print(f"  Capped to {CAP}")

    # Write
    with open(args.out, "w") as f:
        for cid in selected_list:
            f.write(cid + "\n")
    print(f"\nWrote {len(selected_list)} IDs to {args.out}")


if __name__ == "__main__":
    main()
