#!/usr/bin/env python3
"""Single entry point: produce the top-100 submission CSV from candidates.jsonl.

    python rank.py --candidates data/candidates.jsonl --out output/submission.csv

Pure-stdlib, CPU-only, no network — designed to sit inside the challenge's
5-minute / 16 GB ranking budget. (Any embedding precompute happens offline and
is loaded as a cached artifact, not computed here.)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.reasoning import build_reasoning  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402

TOP_N = 100
HEADER = ["candidate_id", "rank", "score", "reasoning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="output/submission.csv")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--rerank-window", type=int, default=200,
                    help="LLM re-ranks the top-N of the ensemble (cached, offline)")
    ap.add_argument("--no-rerank", action="store_true",
                    help="disable LLM re-rank even if a cache is present")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates_list(args.candidates)
    print(f"  {len(candidates):,} candidates loaded in {time.time()-t0:.1f}s")

    scored, ref = score_all(candidates)
    print(f"  scored {len(scored):,} (honeypots dropped); reference date = {ref}")

    # Optional final stage: LLM re-rank of the top window (cached, no network).
    if not args.no_rerank:
        grades = load_grades()
        if grades:
            before = scored[: args.rerank_window]
            scored = apply_rerank(scored, window=args.rerank_window, grades=grades)
            after = scored[: args.rerank_window]
            moved = sum(1 for a, b in zip(before, after) if a["candidate_id"] != b["candidate_id"])
            print(f"  LLM re-rank applied: {len(grades)} grades cached, "
                  f"{moved}/{len(before)} of top-{args.rerank_window} re-ordered")

    top = scored[: args.top]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for rank, row in enumerate(top, start=1):
            reasoning = build_reasoning(row["candidate"], ref, rank)
            w.writerow([row["candidate_id"], rank, row["score"], reasoning])

    print(f"Wrote {len(top)} rows to {out_path} in {time.time()-t0:.1f}s total")


def load_candidates_list(path):
    return list(load_candidates(path))


if __name__ == "__main__":
    main()
