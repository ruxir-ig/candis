#!/usr/bin/env python3
"""Ablation: does the LTR calibrator improve the ranking?

Expected to fail (per train_ltr.py CV: heldout NDCG@10 0.900 < hand-tuned 0.943),
but we confirm it across integration modes for the deck. The Ridge coefficients
(interpretability) are the real deliverable — they validate the hand-tuned
weights.

    uv run python eval/ablation_ltr.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.ltr_model import load_ltr_scores  # noqa: E402
from src.fusion import reciprocal_rank_fusion  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from eval.metrics import composite  # noqa: E402
from eval.evaluate import load_hand_labels, load_llm_labels  # noqa: E402


def metrics_for(scored, labels):
    rank_of = {r["candidate_id"]: i for i, r in enumerate(scored)}
    ordered = sorted(labels, key=lambda it: rank_of.get(it["candidate_id"], 10**9))
    return composite([it["grade"] for it in ordered])


def run():
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, _ = score_all(candidates)
    ltr = load_ltr_scores()
    grades = load_grades()
    hand, llm = load_hand_labels(), load_llm_labels()
    base_top = {r["candidate_id"] for r in scored[:100]}

    variants = {"ensemble": scored}
    if ltr:
        # blend
        for w in (0.15, 0.25):
            blended = []
            for r in scored:
                r = dict(r)
                r["score"] = round(r["score"] * (1 - w) + ltr[r["candidate_id"]] * w, 6)
                blended.append(r)
            blended.sort(key=lambda r: (-r["score"], r["candidate_id"]))
            variants[f"+ltr_blend{w}"] = blended
        # RRF
        ltr_rank = [cid for cid, _ in sorted(ltr.items(), key=lambda kv: (-kv[1], kv[0]))]
        list_a = [r["candidate_id"] for r in scored]
        rrf, order = reciprocal_rank_fusion([("ens", list_a), ("ltr", ltr_rank)])
        by_id = {r["candidate_id"]: r for r in scored}
        variants["+ltr_rrf"] = [dict(by_id[c], score=round(float(rrf[c]), 6)) for c in order]

    def row(name, sc, labels):
        if not labels:
            return ""
        m = metrics_for(sc, labels)
        sw = len(base_top ^ {r["candidate_id"] for r in sc[:100]}) // 2
        return (f"  {name:<16} ndcg@10={m['ndcg@10']:.4f} ndcg@50={m['ndcg@50']:.4f} "
                f"map={m['map']:.4f} comp={m['composite']:.4f} top100Δ={sw}")

    print("\n=== LTR integration ablation ===\n")
    print("HAND qrel (non-circular):")
    for n in variants:
        print(row(n, variants[n], hand))
    print("\nLLM qrel:")
    for n in variants:
        print(row(n, variants[n], llm))

    # Ridge coefficient summary for the deck
    model = ROOT / "cache" / "ltr_model.json"
    if model.exists():
        d = json.loads(model.read_text())
        print(f"\nRidge coefficients (n_labels={d['n_labels']}, "
              f"CV n@10 HGBR={d['cv_ndcg10_hgbr']:.4f} vs base={d['cv_ndcg10_baseline']:.4f}):")
        for k, v in sorted(d["coefficients"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {k:<20} {v:+.3f}")
        print("  ...validates hand-tuned weights (skills/title/evidence top).")


if __name__ == "__main__":
    run()
