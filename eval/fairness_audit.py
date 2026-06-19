"""Fairness / Responsible AI audit for the final top-100 ranking.

Checks whether the ranking is overly dependent on proxy signals (education tier,
company brand, location, experience band, availability) rather than job-relevant
evidence.

Usage:
  uv run python eval/fairness_audit.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.llm_reranker import apply_rerank, load_grades  # noqa: E402
from src.llm_expansion import load_expansion_grades, apply_expansion  # noqa: E402
from src.evidence_graph import evidence_graph_score  # noqa: E402
from src.config import FIT_WEIGHTS_HYBRID, FIT_WEIGHTS  # noqa: E402
import json

SUBMISSION = ROOT / "output" / "submission.csv"
TOP_N = 100


def apply_final_stages(scored):
    grades = load_grades()
    if grades:
        scored = apply_rerank(scored, window=200, grades=grades)
    exp = load_expansion_grades()
    if exp:
        ev_map = {r["candidate_id"]: evidence_graph_score(r["candidate"])["score"]
                  for r in scored}
        base_full = {}
        for line in open(ROOT / "cache" / "llm_rerank.jsonl", encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("fit_score") is not None:
                base_full[d["candidate_id"]] = d
        scored, _ = apply_expansion(scored, base_full, exp, lambda cid: ev_map.get(cid, 0.0))
    return scored


def _tier(c):
    edu = c.get("education") or []
    return edu[0].get("tier", "unknown") if edu else "unknown"


def _company_type(c):
    prof = c.get("profile", {})
    co = (prof.get("current_company") or "").lower()
    big_tech = {"google", "microsoft", "amazon", "meta", "apple", "netflix",
                "uber", "linkedin", "adobe", "salesforce"}
    unicorn = {"swiggy", "zomato", "flipkart", "paytm", "razorpay", "phonepe",
               "freshworks", "dream11", "byju", "unacademy", "curefit", "glance",
               "inmobi", "meesho", "pharmeasy"}
    services = {"tcs", "infosys", "wipro", "accenture", "cognizant",
                "capgemini", "hcl", "tech mahindra", "ibm", "deloitte"}
    if any(b in co for b in big_tech):
        return "big_tech"
    if any(u in co for u in unicorn):
        return "unicorn"
    if any(s in co for s in services):
        return "services"
    return "other_product"


def _location_zone(c):
    loc = (c.get("profile", {}).get("location") or "").lower()
    if "bengaluru" in loc or "bangalore" in loc:
        return "Bangalore"
    if "hyderabad" in loc:
        return "Hyderabad"
    if "pune" in loc:
        return "Pune"
    if "noida" in loc or "gurgaon" in loc or "gurugram" in loc:
        return "Delhi NCR"
    if "mumbai" in loc:
        return "Mumbai"
    if "chennai" in loc:
        return "Chennai"
    if "india" in loc:
        return "Other India"
    return "Non-India"


def _exp_band(c):
    yoe = c.get("profile", {}).get("years_of_experience") or 0
    if yoe < 5:
        return "<5y"
    if yoe <= 9:
        return "5-9y"
    if yoe <= 12:
        return "9-12y"
    return ">12y"


def _availability(c, ref):
    from src.scorers.behavioral_scorer import behavioral_availability
    return behavioral_availability(c, ref)


def _fit(c, ref):
    from src.ranker import score_candidate
    from src.semantic_matcher import semantic_scores
    sem = semantic_scores() or {}
    weights = FIT_WEIGHTS_HYBRID if sem else FIT_WEIGHTS
    return score_candidate(c, ref, weights, sem)


def main():
    print("Loading candidates ...")
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    by_id = {c["candidate_id"]: c for c in candidates}

    # Load submission top-100
    sub_ids = [r["candidate_id"] for r in csv.DictReader(open(SUBMISSION))]
    top100 = [by_id[cid] for cid in sub_ids if cid in by_id]

    # Full pool for comparison
    scored, ref = score_all(candidates)
    scored = apply_final_stages(scored)
    pool = [r["candidate"] for r in scored]

    print(f"  Top-100: {len(top100)}  Pool: {len(pool)}")

    # === 1. Education concentration ===
    print("\n" + "=" * 70)
    print("1. EDUCATION TIER DISTRIBUTION")
    print("=" * 70)
    top_tiers = Counter(_tier(c) for c in top100)
    pool_tiers = Counter(_tier(c) for c in pool)
    print(f"\n  {'Tier':<12} {'Top-100':>8} {'Pool':>8} {'Enrichment':>12}")
    print("  " + "-" * 42)
    for tier in ["tier_1", "tier_2", "tier_3", "unknown"]:
        t_pct = top_tiers.get(tier, 0) / len(top100) * 100
        p_pct = pool_tiers.get(tier, 0) / len(pool) * 100
        enrich = t_pct / p_pct if p_pct else 0
        flag = " ***" if enrich > 2.0 else ""
        print(f"  {tier:<12} {top_tiers.get(tier,0):>8} {pool_tiers.get(tier,0):>8} "
              f"{enrich:>11.1f}x{flag}")

    # === 2. Company-brand concentration ===
    print("\n" + "=" * 70)
    print("2. COMPANY-BRAND DISTRIBUTION")
    print("=" * 70)
    top_cos = Counter(_company_type(c) for c in top100)
    pool_cos = Counter(_company_type(c) for c in pool)
    print(f"\n  {'Type':<16} {'Top-100':>8} {'Pool':>8} {'Enrichment':>12}")
    print("  " + "-" * 46)
    for ct in ["big_tech", "unicorn", "other_product", "services"]:
        t_pct = top_cos.get(ct, 0)
        p_pct = pool_cos.get(ct, 0) / len(pool) * 100
        enrich = (t_pct / len(top100)) / (pool_cos.get(ct, 0) / len(pool)) if pool_cos.get(ct, 0) else 0
        print(f"  {ct:<16} {t_pct:>8} {pool_cos.get(ct,0):>8} {enrich:>11.1f}x")

    # === 3. Location concentration ===
    print("\n" + "=" * 70)
    print("3. LOCATION DISTRIBUTION")
    print("=" * 70)
    top_locs = Counter(_location_zone(c) for c in top100)
    print(f"\n  {'Zone':<16} {'Count':>6} {'%':>6}")
    print("  " + "-" * 30)
    for zone, cnt in top_locs.most_common():
        print(f"  {zone:<16} {cnt:>6} {cnt/len(top100)*100:>5.0f}%")

    # === 4. Experience distribution ===
    print("\n" + "=" * 70)
    print("4. EXPERIENCE DISTRIBUTION")
    print("=" * 70)
    top_exp = Counter(_exp_band(c) for c in top100)
    print(f"\n  {'Band':<8} {'Count':>6} {'%':>6}")
    print("  " + "-" * 22)
    for band in ["<5y", "5-9y", "9-12y", ">12y"]:
        cnt = top_exp.get(band, 0)
        print(f"  {band:<8} {cnt:>6} {cnt/len(top100)*100:>5.0f}%")

    # === 5. Availability influence ===
    print("\n" + "=" * 70)
    print("5. AVAILABILITY INFLUENCE (does it rescue weak fit?)")
    print("=" * 70)
    fits = []
    avails = []
    for c in top100:
        s = _fit(c, ref)
        fits.append(s["fit"])
        avails.append(s["availability"])
    # Correlation check: are low-fit candidates propped up by high availability?
    import statistics
    med_fit = statistics.median(fits)
    low_fit = [(f, a) for f, a in zip(fits, avails) if f < med_fit]
    high_fit = [(f, a) for f, a in zip(fits, avails) if f >= med_fit]
    avg_av_low = statistics.mean(a for _, a in low_fit) if low_fit else 0
    avg_av_high = statistics.mean(a for _, a in high_fit) if high_fit else 0
    print(f"\n  Median fit in top-100: {med_fit:.4f}")
    print(f"  Avg availability (below-median fit): {avg_av_low:.4f}")
    print(f"  Avg availability (above-median fit): {avg_av_high:.4f}")
    print(f"  Difference: {avg_av_low - avg_av_high:+.4f}")
    if avg_av_low > avg_av_high + 0.05:
        print("  *** WARNING: Low-fit candidates have higher availability — possible proxy effect")
    else:
        print("  OK: No evidence availability rescues weak fit")

    # === 6. Component sensitivity ===
    print("\n" + "=" * 70)
    print("6. COMPONENT SENSITIVITY (would removing a component change top-100?)")
    print("=" * 70)
    # Recompute fit without each component
    from src.ranker import score_candidate
    from src.semantic_matcher import semantic_scores
    sem = semantic_scores() or {}
    base_w = dict(FIT_WEIGHTS_HYBRID) if sem else dict(FIT_WEIGHTS)

    for remove in ["education", "location", "experience"]:
        w = {k: v for k, v in base_w.items() if k != remove}
        # Renormalize
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        reranked = sorted(
            (score_candidate(c, ref, w, sem) for c in pool),
            key=lambda r: r["score"], reverse=True
        )
        new_top100_ids = {r["candidate_id"] for r in reranked[:TOP_N]}
        old_top100_ids = set(sub_ids)
        overlap = len(new_top100_ids & old_top100_ids)
        changed = TOP_N - overlap
        print(f"\n  Remove {remove}: {changed} seats change ({overlap}/100 stay)")

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    tier1_pct = top_tiers.get("tier_1", 0) / len(top100) * 100
    bigtech_pct = top_cos.get("big_tech", 0) / len(top100) * 100
    blr_pct = top_locs.get("Bangalore", 0) / len(top100) * 100
    print(f"  Tier-1 education: {tier1_pct:.0f}% of top-100")
    print(f"  Big-tech companies: {bigtech_pct:.0f}% of top-100")
    print(f"  Bangalore-based: {blr_pct:.0f}% of top-100")
    print(f"  Education weight in ranker: {FIT_WEIGHTS_HYBRID.get('education', 0):.0%}")


if __name__ == "__main__":
    main()
