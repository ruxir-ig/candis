#!/usr/bin/env python3
"""Ablation: does the graph-embedding (GNN) ranker improve the ranking?

Graph-only CV NDCG@10 (0.9135) is well below the hand-tuned ensemble (0.9798),
and the adjacency co-occurrence is 98.7% captured by the structured scorers —
so we expect integration to dilute. This confirms it for the deck.

    uv run python eval/ablation_gnn.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.gnn_scores import load_gnn_scores  # noqa: E402
from src.fusion import reciprocal_rank_fusion  # noqa: E402
from eval.metrics import composite  # noqa: E402
from eval.evaluate import load_hand_labels, load_llm_labels  # noqa: E402


def metrics_for(scored, labels):
    rank_of = {r["candidate_id"]: i for i, r in enumerate(scored)}
    ordered = sorted(labels, key=lambda it: rank_of.get(it["candidate_id"], 10**9))
    return composite([it["grade"] for it in ordered])


def run():
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, _ = score_all(candidates)
    gnn = load_gnn_scores()
    hand, llm = load_hand_labels(), load_llm_labels()
    base_top = {r["candidate_id"] for r in scored[:100]}

    variants = {"ensemble": scored}
    if gnn:
        for w in (0.15, 0.25):
            blended = []
            for r in scored:
                r = dict(r)
                r["score"] = round(r["score"] * (1 - w) + gnn[r["candidate_id"]] * w, 6)
                blended.append(r)
            blended.sort(key=lambda r: (-r["score"], r["candidate_id"]))
            variants[f"+gnn_blend{w}"] = blended
        gnn_rank = [cid for cid, _ in sorted(gnn.items(), key=lambda kv: (-kv[1], kv[0]))]
        list_a = [r["candidate_id"] for r in scored]
        rrf, order = reciprocal_rank_fusion([("ens", list_a), ("gnn", gnn_rank)],
                                            weights={"gnn": 0.5})
        by_id = {r["candidate_id"]: r for r in scored}
        variants["+gnn_rrf"] = [dict(by_id[c], score=round(float(rrf[c]), 6)) for c in order]

    def row(name, sc, labels):
        if not labels:
            return ""
        m = metrics_for(sc, labels)
        sw = len(base_top ^ {r["candidate_id"] for r in sc[:100]}) // 2
        return (f"  {name:<16} ndcg@10={m['ndcg@10']:.4f} ndcg@50={m['ndcg@50']:.4f} "
                f"map={m['map']:.4f} comp={m['composite']:.4f} top100Δ={sw}")

    print("\n=== GNN (graph embedding) integration ablation ===\n")
    print("HAND qrel (non-circular):")
    for n in variants:
        print(row(n, variants[n], hand))
    print("\nLLM qrel:")
    for n in variants:
        print(row(n, variants[n], llm))

    summary = ROOT / "cache" / "gnn_summary.json"
    if summary.exists():
        d = json.loads(summary.read_text())
        print(f"\nGraph model: {d['model']}")
        print(f"  graph-only CV NDCG@10 = {d['cv_ndcg10_mean']:.4f} ± {d['cv_ndcg10_std']:.4f} "
              f"(ensemble = 0.9798 on same set)")
        print(f"  top-100 overlap across {len(d['cv_ndcg10_by_seed'])} seeds: "
              f"min {d['top100_overlap_min']}/100 (stable)")
        print(f"  explained variance: {d['explained_variance']:.2%}")


if __name__ == "__main__":
    run()
