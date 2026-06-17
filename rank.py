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

TOP_N = 100
HEADER = ["candidate_id", "rank", "score", "reasoning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="output/submission.csv")
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates_list(args.candidates)
    print(f"  {len(candidates):,} candidates loaded in {time.time()-t0:.1f}s")

    scored, ref = score_all(candidates)
    print(f"  scored {len(scored):,} (honeypots dropped); reference date = {ref}")

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
