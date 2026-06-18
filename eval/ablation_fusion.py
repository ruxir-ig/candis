#!/usr/bin/env python3
"""Ablation: does adding BM25 sparse retrieval (via RRF or weighted blend)
improve the ranking over the rule+semantic ensemble?

Produces one row per variant, measured on BOTH qrels:
  - hand (53 labels, non-circular, near-saturated)
  - llm  (200 labels, sensitive in the mushy middle, partly circular w/ rerank)

Variants:
  1. ensemble           rule+semantic (no rerank, no BM25)      [baseline]
  2. +bm25_weighted     ensemble blended with normalised BM25
  3. +bm25_rrf          ensemble + BM25 via Reciprocal Rank Fusion
  4. +rerank            ensemble + cached LLM re-rank           [current system]
  5. +bm25_rrf +rerank  RRF then LLM re-rank

Acceptance (new_plan.md): NDCG@10 must not drop, top-20 must stay clean, and no
keyword stuffers may enter the top 100. RRF is the recommended integration.

    uv run python eval/ablation_fusion.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all, load_sparse_scores, apply_rrf  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.fusion import rescale_to_unit  # noqa: E402
from eval.metrics import composite  # noqa: E402
from eval.evaluate import load_hand_labels, load_llm_labels  # noqa: E402

BM25_BLEND_WEIGHT = 0.12  # weight on normalised BM25 in the weighted-blend variant


def metrics_for(scored, labels):
    """Order `labels` by the scored list's order, return composite metrics."""
    score_of = {r["candidate_id"]: (len(scored) - i) for i, r in enumerate(scored)}
    ordered = sorted(labels, key=lambda it: score_of.get(it["candidate_id"], -1), reverse=True)
    return composite([it["grade"] for it in ordered])


def top100(scored):
    return [r["candidate_id"] for r in scored[:100]]


def blend_weighted(scored, sparse_norm, w):
    """Simple score blend: final*(1-w) + bm25*w, re-sorted."""
    out = []
    for r in scored:
        r = dict(r)
        r["score"] = round(r["score"] * (1 - w) + sparse_norm.get(r["candidate_id"], 0.0) * w, 6)
        out.append(r)
    out.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return out


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-weight", type=float, default=1.0,
                    help="RRF vote weight for BM25")
    args = ap.parse_args()

    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, ref = score_all(candidates)

    sparse = load_sparse_scores()
    grades = load_grades()

    # Build all variants once.
    variants = {"ensemble": scored}
    if sparse:
        sparse_norm = rescale_to_unit(sparse)
        variants["+bm25_weighted"] = blend_weighted(scored, sparse_norm, BM25_BLEND_WEIGHT)
        variants["+bm25_rrf"] = apply_rrf(scored, sparse, sparse_weight=args.sparse_weight)
    if grades:
        variants["+rerank"] = apply_rerank([dict(r) for r in scored], window=200, grades=grades)
        if sparse:
            variants["+bm25_rrf +rerank"] = apply_rerank(
                [dict(r) for r in variants["+bm25_rrf"]], window=200, grades=grades)

    # Baseline top-100 membership for the "seats changed" stat.
    base_top = set(top100(scored))

    hand = load_hand_labels()
    llm = load_llm_labels()

    def fmt_row(name, scored_list, labels, label_src):
        if not labels:
            return f"  {name:<22} (no {label_src} labels)"
        m = metrics_for(scored_list, labels)
        top = set(top100(scored_list))
        swapped = len(base_top ^ top) // 2
        return (f"  {name:<22} ndcg@10={m['ndcg@10']:.4f} ndcg@50={m['ndcg@50']:.4f} "
                f"map={m['map']:.4f} p@10={m['p@10']:.4f} comp={m['composite']:.4f} "
                f"top100Δ={swapped}")

    print(f"\n=== BM25 / RRF fusion ablation  (sparse_weight={args.sparse_weight}) ===\n")
    print("HAND qrel (non-circular, 53 labels):")
    for name in ("ensemble", "+bm25_weighted", "+bm25_rrf", "+rerank", "+bm25_rrf +rerank"):
        if name in variants:
            print(fmt_row(name, variants[name], hand, "hand"))
    print("\nLLM qrel (200 labels):")
    for name in ("ensemble", "+bm25_weighted", "+bm25_rrf", "+rerank", "+bm25_rrf +rerank"):
        if name in variants:
            print(fmt_row(name, variants[name], llm, "llm"))

    # RRF top-100 diff against ensemble: who swapped?
    if "+bm25_rrf" in variants:
        rrf_top = top100(variants["+bm25_rrf"])
        entered = [c for c in rrf_top if c not in base_top]
        left = [c for c in base_top if c not in set(rrf_top)]
        by_id = {r["candidate_id"]: r for r in scored}
        print(f"\nRRF top-100 churn: +{len(entered)} in / -{len(left)} out")
        for cid in entered[:8]:
            p = by_id[cid]["candidate"]["profile"]
            print(f"    IN  {cid}  {p.get('current_title','')[:30]:30} @ {p.get('current_company','')[:16]}")
        for cid in left[:8]:
            p = by_id[cid]["candidate"]["profile"]
            print(f"    OUT {cid}  {p.get('current_title','')[:30]:30} @ {p.get('current_company','')[:16]}")


if __name__ == "__main__":
    run()
