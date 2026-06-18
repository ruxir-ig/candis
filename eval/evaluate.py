"""Offline evaluation against a labeled candidate set (the qrel).

Two qrel sources, picked via --qrel (default: auto):
  - llm : cache/llm_labels.jsonl  (LLM-judge grades, ~200, de-polarized)
  - hand: eval/golden_labels.csv  (hand grades, 53, polarized / saturated)
  - auto: llm if it has enough valid grades, else hand.

The LLM set is the meaningful one: it covers the mushy middle (grade 2-3) the
hand-set lacks, so the metric finally becomes sensitive enough to tune against.
Hand labels remain useful as a small, high-trust sanity check.

What it measures: take the qrel candidates, order them by the ranker's score,
and compute NDCG@10/@50, MAP, P@10 over that ordering. Honeypots you labeled
that the ranker excluded are treated as ranked last — exactly the behavior we
want to reward.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from eval.metrics import composite  # noqa: E402

HAND_LABELS = ROOT / "eval/golden_labels.csv"
LLM_LABELS = ROOT / "cache/llm_labels.jsonl"

# Minimum valid grades for the LLM qrel to be considered usable in auto mode.
MIN_LLM = 50


def load_hand_labels():
    if not HAND_LABELS.exists():
        return []
    rows = []
    with open(HAND_LABELS) as f:
        for r in csv.DictReader(f):
            g = (r.get("grade") or "").strip()
            if g == "":
                continue
            try:
                grade = int(float(g))
            except ValueError:
                continue
            rows.append({"candidate_id": r["candidate_id"], "grade": grade,
                         "tag": r.get("tag", ""), "source": "hand"})
    return rows


def load_llm_labels():
    if not LLM_LABELS.exists():
        return []
    rows = []
    with open(LLM_LABELS) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("grade") is None:
                continue
            rows.append({"candidate_id": d["candidate_id"], "grade": int(d["grade"]),
                         "tag": "llm", "source": "llm",
                         "rank_when_sampled": d.get("rank_when_sampled")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrel", default="auto", choices=["auto", "llm", "hand"])
    ap.add_argument("--no-rerank", action="store_true",
                    help="measure the rule+semantic ensemble WITHOUT the LLM re-rank")
    args = ap.parse_args()

    llm = load_llm_labels()
    hand = load_hand_labels()
    if args.qrel == "llm":
        labels = llm
        src = "llm"
    elif args.qrel == "hand":
        labels = hand
        src = "hand"
    else:
        if len(llm) >= MIN_LLM:
            labels, src = llm, "llm"
        else:
            labels, src = hand, "hand"
    if not labels:
        print(f"No grades found (hand={len(hand)}, llm={len(llm)}).")
        return

    print(f"Scoring full pool ...  (qrel = {src}: {len(labels)} labels; "
          f"hand={len(hand)} llm={len(llm)})")
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, _ = score_all(candidates)
    # Mirror rank.py: apply the cached LLM re-rank so we measure what we submit.
    grades = load_grades()
    if grades and not args.no_rerank:
        scored = apply_rerank(scored, window=200, grades=grades)
        print(f"  (LLM re-rank applied: {len(grades)} grades cached)")
    elif args.no_rerank:
        print("  (--no-rerank: measuring rule+semantic ensemble only)")
    rank_of = {r["candidate_id"]: i + 1 for i, r in enumerate(scored)}
    score_of = {r["candidate_id"]: r["score"] for r in scored}
    title_of = {
        r["candidate_id"]: r["candidate"]["profile"].get("current_title", "")
        for r in scored
    }

    def sort_key(item):
        return score_of.get(item["candidate_id"], float("-inf"))
    ordered = sorted(labels, key=sort_key, reverse=True)
    rels = [item["grade"] for item in ordered]

    m = composite(rels)
    dist = {g: sum(1 for it in labels if it["grade"] == g) for g in range(5)}

    print(f"\nGrade distribution {dict(sorted(dist.items()))}")
    print("-" * 60)
    for k in ("ndcg@10", "ndcg@50", "map", "p@10", "p@5", "composite"):
        print(f"  {k:>10}: {m[k]:.4f}")
    print("-" * 60)

    # Disagreements: good candidates buried, weak ones surfaced.
    print("\nBiggest model/judgment disagreements:")
    flagged = []
    for it in labels:
        cid, grade = it["candidate_id"], it["grade"]
        sysrank = rank_of.get(cid)
        if grade >= 3 and (sysrank is None or sysrank > 100):
            flagged.append((cid, grade, sysrank, it.get("tag", ""), "GOOD but buried"))
        elif grade <= 1 and sysrank is not None and sysrank <= 100:
            flagged.append((cid, grade, sysrank, it.get("tag", ""), "WEAK but in top 100"))
    if not flagged:
        print("  (none — model and labels broadly agree on the extremes)")
    for cid, grade, sysrank, tag, why in flagged[:15]:
        sr = "excluded" if sysrank is None else f"#{sysrank}"
        print(f"  {cid}  grade={grade}  sys={sr:>9}  [{tag}]  "
              f"{title_of.get(cid,'?')[:34]}  <- {why}")


if __name__ == "__main__":
    main()
