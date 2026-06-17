"""Offline evaluation against your hand-labeled golden set.

Workflow:
  1. python eval/build_golden_set.py     # generates the labeling sheet
  2. fill grades (0-4) in eval/golden_labels.csv
  3. python eval/evaluate.py              # prints metrics + disagreements

What it measures: take the candidates you labeled, order them by the ranker's
score, and compute NDCG@10/@50, MAP, P@10 over that ordering. This is a *proxy*
for the hidden ground truth (small sample, your judgment), but it's a real,
repeatable signal — change a weight, re-run, see if the proxy moves.

Honeypots you labeled (grade 0) that the ranker excluded are treated as ranked
last — which is exactly the behavior we want to reward.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from eval.metrics import composite  # noqa: E402

LABELS = ROOT / "eval/golden_labels.csv"


def load_labels():
    if not LABELS.exists():
        return []
    rows = []
    with open(LABELS) as f:
        for r in csv.DictReader(f):
            g = (r.get("grade") or "").strip()
            if g == "":
                continue
            try:
                grade = int(float(g))
            except ValueError:
                continue
            rows.append((r["candidate_id"], grade, r.get("tag", ""), r.get("system_rank", "")))
    return rows


def main():
    labels = load_labels()
    if not labels:
        print(f"No grades found in {LABELS}.")
        print("Run `python eval/build_golden_set.py`, fill the 'grade' column (0-4), then re-run.")
        return

    print("Scoring full pool ...")
    candidates = list(load_candidates(ROOT / "data/candidates.jsonl"))
    scored, _ = score_all(candidates)
    rank_of = {r["candidate_id"]: i + 1 for i, r in enumerate(scored)}
    score_of = {r["candidate_id"]: r["score"] for r in scored}
    title_of = {
        r["candidate_id"]: r["candidate"]["profile"].get("current_title", "")
        for r in scored
    }

    # Order labeled candidates by the ranker's score; excluded (honeypot) -> last.
    def sort_key(item):
        cid = item[0]
        return score_of.get(cid, float("-inf"))

    ordered = sorted(labels, key=sort_key, reverse=True)
    rels = [grade for _, grade, _, _ in ordered]

    m = composite(rels)
    dist = {g: sum(1 for _, gr, _, _ in labels if gr == g) for g in range(5)}

    print(f"\nLabeled: {len(labels)}   grade distribution {dict(sorted(dist.items()))}")
    print("-" * 60)
    for k in ("ndcg@10", "ndcg@50", "map", "p@10", "p@5", "composite"):
        print(f"  {k:>10}: {m[k]:.4f}")
    print("-" * 60)

    # Disagreements worth investigating: high grade but buried, or low grade but
    # ranked high. These are where the model and your judgment diverge.
    print("\nBiggest model/judgment disagreements:")
    flagged = []
    for cid, grade, tag, _ in labels:
        sysrank = rank_of.get(cid)  # None if excluded as honeypot
        # crude divergence: good candidate ranked deep, or bad one ranked shallow
        if grade >= 3 and (sysrank is None or sysrank > 100):
            flagged.append((cid, grade, sysrank, tag, "GOOD but buried"))
        elif grade <= 1 and sysrank is not None and sysrank <= 100:
            flagged.append((cid, grade, sysrank, tag, "WEAK but in top 100"))
    if not flagged:
        print("  (none — model and labels broadly agree on the extremes)")
    for cid, grade, sysrank, tag, why in flagged[:15]:
        sr = "excluded" if sysrank is None else f"#{sysrank}"
        print(f"  {cid}  grade={grade}  sys={sr:>9}  [{tag}]  {title_of.get(cid,'?')}  <- {why}")


if __name__ == "__main__":
    main()
