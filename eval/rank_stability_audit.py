"""Rank stability audit — how robust is the final ranking to cache/model deps?

Runs the ranker in 4 degraded modes and compares top-K overlap + hand qrel
metrics against the default submission. Answers: is the system overdependent
on one cache or model?

Modes (pre-computed CSVs):
  full         : output/submission.csv (ensemble + rerank + expansion)
  no_expansion : ensemble + rerank (no expansion)
  base_ensemble: ensemble only (no rerank, no expansion)
  rule_only    : rules only (no semantic, no rerank, no expansion)

Usage:
  # First generate the mode CSVs:
  uv run python rank.py --out /tmp/stability/no_expansion.csv --no-expansion
  uv run python rank.py --out /tmp/stability/base_ensemble.csv --no-rerank --no-expansion
  mv cache/embeddings.npz cache/embeddings.npz.bak
  uv run python rank.py --out /tmp/stability/rule_only.csv --no-rerank --no-expansion
  mv cache/embeddings.npz.bak cache/embeddings.npz

  # Then run this audit:
  uv run python eval/rank_stability_audit.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import ndcg_at_k, average_precision, precision_at_k  # noqa: E402

FULL = ROOT / "output" / "submission.csv"
STABILITY_DIR = Path("/tmp/stability")

MODES = {
    "full (default)": FULL,
    "no_expansion": STABILITY_DIR / "no_expansion.csv",
    "base_ensemble": STABILITY_DIR / "no_rerank.csv",
    "rule_only": STABILITY_DIR / "rule_only.csv",
}


def load_ids(path: Path) -> list[str]:
    """Load candidate IDs in rank order from a submission CSV."""
    rows = list(csv.DictReader(open(path)))
    return [r["candidate_id"] for r in rows]


def overlap(a: list[str], b: list[str], k: int) -> float:
    """Fraction of top-k IDs shared between two rankings."""
    sa = set(a[:k])
    sb = set(b[:k])
    return len(sa & sb) / k if k else 0.0


def load_hand_qrel():
    """Load hand labels as {candidate_id: relevance_grade}."""
    labels = {}
    p = ROOT / "eval" / "golden_labels.csv"
    if not p.exists():
        return labels
    for row in csv.DictReader(open(p)):
        cid = row.get("candidate_id", "")
        grade = row.get("grade") or row.get("relevance")
        if cid and grade is not None:
            try:
                labels[cid] = int(grade)
            except (ValueError, TypeError):
                pass
    return labels


def evaluate_mode(ids: list[str], qrel: dict) -> dict:
    """Compute ranking metrics for one mode's ordering against the qrel."""
    rels = [qrel.get(cid, 0) for cid in ids]
    return {
        "NDCG@10": round(ndcg_at_k(rels, 10), 4),
        "NDCG@50": round(ndcg_at_k(rels, 50), 4),
        "MAP": round(average_precision(rels), 4),
        "P@10": round(precision_at_k(rels, 10), 4),
    }


def main():
    print("=" * 80)
    print("RANK STABILITY AUDIT")
    print("=" * 80)

    # Load all modes
    mode_ids = {}
    for name, path in MODES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}")
            continue
        mode_ids[name] = load_ids(path)
        print(f"  {name}: {len(mode_ids[name])} candidates")

    if "full (default)" not in mode_ids:
        sys.exit("ERROR: default submission not found")

    full_ids = mode_ids["full (default)"]
    qrel = load_hand_qrel()
    print(f"  Hand qrel: {len(qrel)} labels")

    # --- Top-K overlap with default ---
    print()
    print("-" * 80)
    print("TOP-K OVERLAP WITH DEFAULT SUBMISSION")
    print("-" * 80)
    print(f"  {'Mode':<20} {'Top-20':>8} {'Top-50':>8} {'Top-100':>9}")
    print("  " + "-" * 47)
    for name, ids in mode_ids.items():
        if name == "full (default)":
            continue
        o20 = overlap(full_ids, ids, 20)
        o50 = overlap(full_ids, ids, 50)
        o100 = overlap(full_ids, ids, 100)
        print(f"  {name:<20} {o20:>7.1%} {o50:>7.1%} {o100:>8.1%}")

    # --- Hand qrel metrics ---
    print()
    print("-" * 80)
    print("HAND QREL METRICS BY MODE")
    print("-" * 80)
    print(f"  {'Mode':<20} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>8} {'P@10':>8}")
    print("  " + "-" * 55)
    for name, ids in mode_ids.items():
        m = evaluate_mode(ids, qrel)
        print(f"  {name:<20} {m['NDCG@10']:>8.4f} {m['NDCG@50']:>8.4f} "
              f"{m['MAP']:>8.4f} {m['P@10']:>8.4f}")

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find the most degraded mode
    worst_overlap = 1.0
    worst_mode = ""
    for name, ids in mode_ids.items():
        if name == "full (default)":
            continue
        o100 = overlap(full_ids, ids, 100)
        if o100 < worst_overlap:
            worst_overlap = o100
            worst_mode = name

    print(f"  Most degraded mode: {worst_mode} (top-100 overlap: {worst_overlap:.1%})")

    # Top-20 stability
    if "rule_only" in mode_ids:
        o20_rule = overlap(full_ids, mode_ids["rule_only"], 20)
        print(f"  Top-20 stability (rule-only vs full): {o20_rule:.1%}")

    if "no_expansion" in mode_ids:
        o100_ne = overlap(full_ids, mode_ids["no_expansion"], 100)
        print(f"  Top-100 stability (no-expansion vs full): {o100_ne:.1%}")
        print(f"  (Expansion changes {1-o100_ne:.0%} of top-100 = {100-int(o100_ne*100)} seats)")


if __name__ == "__main__":
    main()
