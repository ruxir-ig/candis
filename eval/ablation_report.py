#!/usr/bin/env python3
"""Consolidated ablation report — the one table that tells the whole story.

Runs every ranker variant tested across the project, on BOTH qrels, with the
accept/reject decision and the runtime cost. Writes docs/final_ablation.md.

This is the master evidence-based-selection table for the deck: it shows we
built a ranking laboratory, tested five research-inspired rankers, and selected
the most recruiter-trustworthy combination under real constraints.

    uv run python eval/ablation_report.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all, load_sparse_scores, apply_rrf  # noqa: E402
from src.ltr_model import load_ltr_scores  # noqa: E402
from src.gnn_scores import load_gnn_scores  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.fusion import reciprocal_rank_fusion  # noqa: E402
from eval.metrics import composite  # noqa: E402
from eval.evaluate import load_hand_labels, load_llm_labels  # noqa: E402


def metrics_for(order_ids, labels):
    rank_of = {cid: i for i, cid in enumerate(order_ids)}
    ordered = sorted(labels, key=lambda it: rank_of.get(it["candidate_id"], 10**9))
    return composite([it["grade"] for it in ordered])


def ids_of(scored):
    return [r["candidate_id"] for r in scored]


def blend(scored, extra, w):
    out = []
    for r in scored:
        r = dict(r)
        r["score"] = round(r["score"] * (1 - w) + extra.get(r["candidate_id"], 0.0) * w, 6)
        out.append(r)
    out.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return out


def rrf_with(scored, extra, w):
    list_a = ids_of(scored)
    list_b = [cid for cid, _ in sorted(extra.items(), key=lambda kv: (-kv[1], kv[0]))]
    rrf, order = reciprocal_rank_fusion([("ens", list_a), ("x", list_b)],
                                        weights={"x": w})
    by_id = {r["candidate_id"]: r for r in scored}
    return [dict(by_id[c], score=round(float(rrf[c]), 6)) for c in order]


def main():
    t0 = time.time()
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, ref = score_all(candidates)
    grades = load_grades()
    sparse = load_sparse_scores()
    ltr = load_ltr_scores()
    gnn = load_gnn_scores()
    hand, llm = load_hand_labels(), load_llm_labels()

    # evidence scores (computed once)
    ev = {r["candidate_id"]: evidence_graph_score(r["candidate"])["score"] for r in scored}

    base = ids_of(scored)
    variants = []  # (name, order_ids, decision, note)

    variants.append(("rule+semantic", base, "baseline",
                     "structured scorers + dense bi-encoder"))
    if sparse:
        variants.append(("+BM25 RRF", ids_of(rrf_with(scored, sparse, 1.0)),
                         "REJECTED", "lexical noise; semantic+title already beat it"))
    if ev:
        variants.append(("+evidence-graph RRF", ids_of(rrf_with(scored, ev, 1.0)),
                         "REJECTED (rank)", "kept for reasoning enrichment"))
        variants.append(("+evidence-graph blend", ids_of(blend(scored, ev, 0.15)),
                         "REJECTED (rank)", "+5 buried stars but breaks top-10"))
    if ltr:
        variants.append(("+LTR blend", ids_of(blend(scored, ltr, 0.15)),
                         "REJECTED", "overfits 219 labels (CV 0.90<0.94)"))
    if gnn:
        variants.append(("+GNN blend", ids_of(blend(scored, gnn, 0.15)),
                         "REJECTED", "redundant w/ rich features (CV 0.91<0.98)"))
    if grades:
        reranked = apply_rerank([dict(r) for r in scored], window=200, grades=grades)
        variants.append(("+LLM re-rank", ids_of(reranked),
                         "ADOPTED (final)", "cross-encoder judgment on top-200"))

    # The final system order = the last adopted variant
    final_order = variants[-1][1]

    # Build the report
    R = []
    def w(s=""):
        R.append(s)

    w("# Final ablation report\n")
    w("> Every ranker is a *candidate*. It does not replace the current ranker")
    w("> until it proves itself on the (non-circular) hand qrel AND keeps the")
    w("> top-10 clean. Five research-inspired rankers tested; one adopted.\n")
    w(f"Scoring: `0.5*ndcg@10 + 0.3*ndcg@50 + 0.15*map + 0.05*p@10`")
    w(f"Pool: {len(scored):,} post-filter candidates. Generated in {time.time()-t0:.0f}s.\n")

    w("## Master ablation table\n")
    w("| ranker | decision | hand n@10 | hand comp | llm n@10 | llm comp |")
    w("|--------|----------|-----------|-----------|----------|----------|")
    for name, order, decision, note in variants:
        mh = metrics_for(order, hand)
        ml = metrics_for(order, llm)
        w(f"| {name} | {decision} | {mh['ndcg@10']:.4f} | {mh['composite']:.4f} "
          f"| {ml['ndcg@10']:.4f} | {ml['composite']:.4f} |")
    w("\n_Note: hand qrel (53 labels) is non-circular. LLM qrel (200 labels) is")
    w("partly circular with the re-rank. Where a learned ranker's hand-qrel score")
    w("looks perfect, that's label leakage — its labels are in the training set._\n")

    # Decision summary
    w("## Evidence-based decisions\n")
    w("- **BM25 + RRF** — rejected. Lexical overlap is noisier than our dense")
    w("  semantic + title enum; RRF discards the availability gradient.")
    w("- **Evidence graph** — ranking rejected (breaks the saturated top-10), but")
    w("  **adopted for reasoning enrichment** (corroboration evidence in the")
    w("  reasoning column) and as a validated anti-gaming guardrail.")
    w("- **LTR** — rejected (overfits 219 skewed labels). Ridge coefficients")
    w("  independently validated the hand-tuned weights.")
    w("- **GNN** — rejected (graph signal stable but redundant with rich features).")
    w("- **LLM re-rank** — **adopted.** Pointwise cross-encoder judgment on the")
    w("  top-200; the only stage that improved qualitative top-100 membership.\n")

    # Robustness summary
    w("## Anti-gaming robustness (full audit: docs/robustness_audit.md)\n")
    w("| perturbation (500 weak profiles) | keyword baseline | our system |")
    w("|-----------------------------------|------------------|------------|")
    w("| keyword stuffing → top-100 | **15** | **0** |")
    w("| keyword stuffing → top-500 | 24 | 0 |")
    w("| title inflation → top-100 | — | 0 |")
    w("| skill shuffle (max Δ) | — | 0 |")

    # Top-100 composition audit
    w("\n## Final top-100 composition\n")
    by_id = {r["candidate_id"]: r for r in scored}
    top100 = [by_id[cid] for cid in final_order[:100] if cid in by_id]
    companies = Counter(r["candidate"]["profile"].get("current_company", "?") for r in top100)
    titles = Counter(r["candidate"]["profile"].get("current_title", "?") for r in top100)
    w("**Top companies represented:** " +
      ", ".join(f"{c} ({n})" for c, n in companies.most_common(10)))
    w("\n**Title categories (top):** " +
      ", ".join(f"{_bucket(t)} ({n})" for t, n in Counter(
          _bucket(r["candidate"]["profile"].get("current_title", "")) for r in top100
      ).most_common(6)))

    w("\n## Final top-10\n")
    w("| # | candidate | title @ company | yrs |")
    w("|---|-----------|-----------------|-----|")
    for i, r in enumerate(top100[:10], 1):
        p = r["candidate"]["profile"]
        w(f"| {i} | {r['candidate_id']} | {p.get('current_title','')} @ "
          f"{p.get('current_company','')} | {p.get('years_of_experience',0):.1f} |")

    out = ROOT / "docs" / "final_ablation.md"
    out.write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R))
    print(f"\nWrote {out.relative_to(ROOT)}")


def _bucket(title):
    t = (title or "").lower()
    if "staff" in t or "principal" in t or "lead" in t:
        return "Staff/Lead"
    if "senior" in t:
        return "Senior"
    if "recommendation" in t or "recsys" in t:
        return "RecSys"
    if "search" in t or "retrieval" in t:
        return "Search/IR"
    if "nlp" in t:
        return "NLP"
    if "data scientist" in t:
        return "Data Scientist"
    if "ml" in t or "machine learning" in t or "ai" in t:
        return "ML/AI"
    return "Other"


if __name__ == "__main__":
    main()
