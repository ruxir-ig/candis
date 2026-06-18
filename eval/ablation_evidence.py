#!/usr/bin/env python3
"""Ablation: does the evidence-graph ranker improve the ranking, and via which
integration mode?

The evidence graph scores *cross-field corroboration* (skills + career text +
assessments + title all agree) — signal no single per-field scorer captures. We
test four ways to fold it into the proven rule+semantic ensemble:

  1. RRF              fuse ensemble + evidence rank lists via reciprocal rank
  2. weighted blend   final*(1-w) + evidence*w
  3. penalty mult     final * (floor + (1-floor)*evidence)   (evidence as a gate
                      that can only dampen a weakly-corroborated candidate,
                      never rescue one — mirrors the availability multiplier)
  4. penalty-only     dampen ONLY when evidence < 0.5, else leave final untouched

Measured on both qrels (hand = non-circular, llm = sensitive middle).

    uv run python eval/ablation_evidence.py
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all, apply_rrf  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from eval.metrics import composite  # noqa: E402
from eval.evaluate import load_hand_labels, load_llm_labels  # noqa: E402


def metrics_for(order_ids, labels):
    """order_ids: full ranked candidate_id list. Returns composite over labels."""
    rank_of = {cid: i for i, cid in enumerate(order_ids)}
    ordered = sorted(labels, key=lambda it: rank_of.get(it["candidate_id"], 10**9))
    return composite([it["grade"] for it in ordered])


def top100_ids(scored):
    return [r["candidate_id"] for r in scored[:100]]


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-w", type=float, default=0.20)
    ap.add_argument("--penalty-floor", type=float, default=0.55)
    ap.add_argument("--rrf-w", type=float, default=1.0)
    args = ap.parse_args()

    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, _ = score_all(candidates)
    base_order = [r["candidate_id"] for r in scored]

    # Compute evidence score for every scored candidate (one pass).
    t0 = time.time()
    ev = {}
    by_id = {r["candidate_id"]: r for r in scored}
    for r in scored:
        ev[r["candidate_id"]] = evidence_graph_score(r["candidate"])["score"]
    print(f"  evidence scored {len(ev):,} candidates in {time.time()-t0:.1f}s")

    ev_rank = [cid for cid, _ in sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))]

    # Build the four variants.
    variants = {"ensemble": scored}

    # 1. RRF
    variants["+ev_rrf"] = apply_rrf(scored, ev, sparse_weight=args.rrf_w) \
        if False else _rrf_ev(scored, ev_rank, args.rrf_w)

    # 2. weighted blend
    w = args.blend_w
    blended = []
    for r in scored:
        r = dict(r)
        r["score"] = round(r["score"] * (1 - w) + ev[r["candidate_id"]] * w, 6)
        blended.append(r)
    blended.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    variants["+ev_blend"] = blended

    # 3. penalty multiplier
    fl = args.penalty_floor
    pen = []
    for r in scored:
        r = dict(r)
        e = ev[r["candidate_id"]]
        r["score"] = round(r["score"] * (fl + (1 - fl) * e), 6)
        pen.append(r)
    pen.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    variants["+ev_penalty"] = pen

    # 4. penalty-only (dampen only when evidence < 0.5)
    penonly = []
    for r in scored:
        r = dict(r)
        e = ev[r["candidate_id"]]
        if e < 0.5:
            r["score"] = round(r["score"] * (fl + (1 - fl) * e), 6)
        penonly.append(r)
    penonly.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    variants["+ev_penalty_only"] = penonly

    hand = load_hand_labels()
    llm = load_llm_labels()

    base_top = set(top100_ids(scored))

    def row(name, scored_list, labels, src):
        if not labels:
            return f"  {name:<20} (no {src} labels)"
        order = [r["candidate_id"] for r in scored_list]
        m = metrics_for(order, labels)
        swapped = len(base_top ^ set(order[:100])) // 2
        return (f"  {name:<20} ndcg@10={m['ndcg@10']:.4f} ndcg@50={m['ndcg@50']:.4f} "
                f"map={m['map']:.4f} comp={m['composite']:.4f} top100Δ={swapped}")

    print(f"\n=== Evidence-graph integration ablation ===\n")
    names = ["ensemble", "+ev_rrf", "+ev_blend", "+ev_penalty", "+ev_penalty_only"]
    print(f"HAND qrel (non-circular, 53 labels)  [blend_w={args.blend_w}, floor={args.penalty_floor}, rrf_w={args.rrf_w}]")
    for n in names:
        print(row(n, variants[n], hand, "hand"))
    print("\nLLM qrel (200 labels):")
    for n in names:
        print(row(n, variants[n], llm, "llm"))

    # Where do the buried grade-4s land under the best-looking variant?
    # Show churn for penalty mode (usually cleanest).
    for label_mode, labels in (("HAND", hand), ("LLM", llm)):
        buried = [(it["candidate_id"], it["grade"]) for it in labels
                  if it["grade"] >= 3]
        print(f"\n{label_mode} grade>=3 candidates: how many reach top-100?")
        for n in names:
            order = [r["candidate_id"] for r in variants[n]]
            in100 = sum(1 for cid, _ in buried if cid in set(order[:100]))
            print(f"   {n:<20} {in100}/{len(buried)} in top-100")


def _rrf_ev(scored, ev_rank, w):
    from src.fusion import reciprocal_rank_fusion
    list_a = [r["candidate_id"] for r in scored]
    rrf, order = reciprocal_rank_fusion(
        [("ensemble", list_a), ("evidence", ev_rank)],
        weights={"evidence": w})
    by_id = {r["candidate_id"]: r for r in scored}
    out = []
    for cid in order:
        r = dict(by_id[cid])
        r["score"] = round(float(rrf[cid]), 6)
        out.append(r)
    return out


if __name__ == "__main__":
    run()
