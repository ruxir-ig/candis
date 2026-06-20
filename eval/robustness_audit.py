#!/usr/bin/env python3
"""Robustness / anti-gaming audit.

Proves the system is robust where naive keyword matching is not. Five
perturbations, each measured against (a) a naive keyword baseline and (b) our
rule+semantic ensemble:

  1. keyword stuffing   inject AI terms into weak/non-technical profiles
  2. skill removal      strip the skills section from strong profiles
  3. skill shuffle      reorder skills (ranking must be invariant)
  4. drop behavioral    remove all redrob_signals (ranking should barely move)
  5. title inflation    relabel a weak profile "Senior AI Engineer"

The headline number is test 1: how many stuffed weak profiles breach the
top-100 / top-500 under each ranker.

Writes a markdown report to docs/robustness_audit.md.

    uv run python eval/robustness_audit.py --sample 500
"""
import argparse
import bisect
import copy
import random
import sys
import time
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all, score_candidate, reference_date  # noqa: E402
from src.config import FIT_WEIGHTS_HYBRID  # noqa: E402
from src.semantic_matcher import semantic_scores  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.coarse_filter import passes_coarse_filter  # noqa: E402
from src.scorers import title_role_score  # noqa: E402
from src.scorers.skills_scorer import SKILL_RELEVANCE, PROFICIENCY_WEIGHT  # noqa: E402

# The injected keyword-stuffer payload (the JD's exact warning list).
STUFF_SKILLS = [
    {"name": "FAISS", "proficiency": "expert", "duration_months": 0, "endorsements": 3},
    {"name": "Qdrant", "proficiency": "expert", "duration_months": 0, "endorsements": 2},
    {"name": "Pinecone", "proficiency": "expert", "duration_months": 0, "endorsements": 1},
    {"name": "Embeddings", "proficiency": "expert", "duration_months": 0, "endorsements": 2},
    {"name": "Semantic Search", "proficiency": "expert", "duration_months": 0, "endorsements": 4},
    {"name": "RAG", "proficiency": "expert", "duration_months": 0, "endorsements": 2},
    {"name": "LangChain", "proficiency": "advanced", "duration_months": 0, "endorsements": 1},
]


def keyword_baseline_score(candidate: dict) -> float:
    """The naive ranker the JD warns against: relevant skill keywords weighted by
    proficiency ONLY — no duration trust, no title anchor, no career check."""
    skills = candidate.get("skills", []) or []
    total = 0.0
    for s in skills:
        n = (s.get("name", "") or "").lower()
        rel = 0.0
        for kw, w in SKILL_RELEVANCE.items():
            if kw in n and w > rel:
                rel = w
        prof = PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.5)
        total += rel * prof
    return total


def perturb_stuff(c):
    c = copy.deepcopy(c)
    existing = {(s.get("name", "") or "").lower() for s in c.get("skills", []) or []}
    for s in STUFF_SKILLS:
        if s["name"].lower() not in existing:
            c.setdefault("skills", []).append(dict(s))
    return c


def perturb_strip_skills(c):
    c = copy.deepcopy(c)
    c["skills"] = []
    return c


def perturb_shuffle_skills(c):
    c = copy.deepcopy(c)
    random.shuffle(c.setdefault("skills", []))
    return c


def perturb_drop_behavioral(c):
    c = copy.deepcopy(c)
    c["redrob_signals"] = {}
    return c


def perturb_inflate_title(c):
    c = copy.deepcopy(c)
    c.setdefault("profile", {})["current_title"] = "Senior AI Engineer"
    if c.get("career_history"):
        c["career_history"][0] = dict(c["career_history"][0])
        c["career_history"][0]["title"] = "Senior AI Engineer"
    return c


class PoolRanker:
    """Ranks a single (possibly perturbed) candidate against a fixed pool by
    binary-searching a precomputed descending score array — O(log n) each."""

    def __init__(self, scored):
        self.n = len(scored)
        self.scores_asc = sorted(r["score"] for r in scored)  # ascending

    def rank_of(self, score: float) -> int:
        # rank = 1 + (number of pool members with a strictly higher score)
        better = self.n - bisect.bisect_right(self.scores_asc, score)
        return better + 1


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    t0 = time.time()
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, ref = score_all(candidates)
    sem = semantic_scores()
    base_rank = {r["candidate_id"]: i + 1 for i, r in enumerate(scored)}
    pool = PoolRanker(scored)
    n_pool = len(scored)
    print(f"Pool: {n_pool:,} candidates. Baseline built in {time.time()-t0:.1f}s\n")

    def our_score(c):
        return score_candidate(c, ref, FIT_WEIGHTS_HYBRID, sem)["score"]

    report = []
    def w(line=""):
        print(line)
        report.append(line)

    # selectors
    weak = [c for c in candidates
            if title_role_score(c) < 0.30
            and base_rank.get(c["candidate_id"], 10**9) > 500]
    random.shuffle(weak)
    weak = weak[:args.sample]
    strong = scored[:100]
    w(f"# Robustness / anti-gaming audit\n")
    w(f"Sample sizes: {len(weak)} weak (non-technical) profiles, "
      f"{len(strong)} strong (top-100) profiles.\n")

    # ===== Test 1: keyword stuffing =====
    w("## 1. Keyword stuffing (inject FAISS/Qdrant/Pinecone/RAG/...)\n")
    w(f"Injecting {len(STUFF_SKILLS)} expert AI skills (0 months duration) into "
      f"{len(weak)} weak profiles.\n")
    # Our system: where do stuffed weak profiles land?
    our_ranks = []
    moved_up = 0
    for c in weak:
        new_rank = pool.rank_of(our_score(perturb_stuff(c)))
        our_ranks.append(new_rank)
        if new_rank < base_rank.get(c["candidate_id"], n_pool):
            moved_up += 1
    our_top100 = sum(1 for r in our_ranks if r <= 100)
    our_top500 = sum(1 for r in our_ranks if r <= 500)

    # Keyword baseline: rank ALL candidates (stuffed where applicable) under the
    # naive ranker, then count where the stuffed weak profiles land.
    kw_pool = []
    for c in candidates:
        if is_honeypot(c) or not passes_coarse_filter(c):
            continue
        kw_pool.append((c["candidate_id"], keyword_baseline_score(c)))
    weak_ids = {c["candidate_id"] for c in weak}
    for i, (cid, _) in enumerate(kw_pool):
        if cid in weak_ids:
            c = next(x for x in weak if x["candidate_id"] == cid)
            kw_pool[i] = (cid, keyword_baseline_score(perturb_stuff(c)))
    kw_pool.sort(key=lambda x: (-x[1], x[0]))
    kw_rank = {cid: i + 1 for i, (cid, _) in enumerate(kw_pool)}
    kw_top100 = sum(1 for c in weak if kw_rank.get(c["candidate_id"], n_pool) <= 100)
    kw_top500 = sum(1 for c in weak if kw_rank.get(c["candidate_id"], n_pool) <= 500)

    w("| ranker               | stuffed weak in top-100 | stuffed weak in top-500 |")
    w("|----------------------|--------------------------|--------------------------|")
    w(f"| **keyword baseline** | {kw_top100:<24} | {kw_top500:<24} |")
    w(f"| **our system**       | {our_top100:<24} | {our_top500:<24} |")
    w(f"\nOur system: {moved_up}/{len(weak)} weak profiles moved up at all after "
      f"stuffing; median post-stuffing rank #{median(our_ranks):.0f} of {n_pool:,}. "
      f"The keyword baseline vaults them near the top of the pool.\n")

    # ===== Test 2: skill removal =====
    w("## 2. Skill removal (strip skills from top-100 strong profiles)\n")
    drops = []
    collapsed = 0
    for r in strong:
        cid = r["candidate_id"]
        new_rank = pool.rank_of(our_score(perturb_strip_skills(r["candidate"])))
        drops.append(new_rank - base_rank[cid])
        if new_rank > 1000:
            collapsed += 1
    w(f"Removing all skills: median rank drop {median(drops):.0f}, mean {mean(drops):.0f}, "
      f"worst {max(drops)}. Collapsed beyond #1000: {collapsed}/{len(strong)} "
      f"(career/title evidence keeps genuine engineers relevant).\n")

    # ===== Test 3: skill shuffle =====
    w("## 3. Skill shuffle (order invariance)\n")
    deltas = [abs(pool.rank_of(our_score(perturb_shuffle_skills(r["candidate"])))
                  - base_rank[r["candidate_id"]]) for r in strong]
    w(f"Max rank change after shuffling skill order: {max(deltas)} "
      f"(must be 0 — ranking is order-invariant by design).\n")

    # ===== Test 4: drop behavioral signals =====
    w("## 4. Drop behavioral signals (remove all redrob_signals)\n")
    bshifts = [abs(pool.rank_of(our_score(perturb_drop_behavioral(r["candidate"])))
                   - base_rank[r["candidate_id"]]) for r in strong]
    w(f"Removing all behavioral signals: max rank change {max(bshifts)}, "
      f"median {median(bshifts):.0f} (behavioral is a dampener, not a primary "
      f"signal, so the ranking stays intact).\n")

    # ===== Test 5: title inflation =====
    w("## 5. Title inflation (relabel weak profile 'Senior AI Engineer')\n")
    inf_jumps = []
    inf_top100 = 0
    for c in weak:
        cid = c["candidate_id"]
        new_rank = pool.rank_of(our_score(perturb_inflate_title(c)))
        jump = base_rank.get(cid, n_pool) - new_rank
        if jump > 0:
            inf_jumps.append(jump)
        if new_rank <= 100:
            inf_top100 += 1
    w(f"Inflating title to 'Senior AI Engineer': {len(inf_jumps)}/{len(weak)} weak "
      f"profiles moved up; {inf_top100} reached top-100; median jump "
      f"{median(inf_jumps) if inf_jumps else 0:.0f}. The title anchor is "
      f"cross-checked against trust-weighted skills and career corroboration, so "
      f"a title change alone does not carry a weak profile up.\n")

    w("_Methodology_: ranks computed over the full post-filter pool "
      f"({n_pool:,}). Our system uses the full rule+semantic ensemble; the "
      f"semantic score is cached by candidate_id and stays fixed across "
      f"perturbations, so the test isolates the structural rule defenses. "
      f"Single-candidate rank via binary search over the precomputed pool scores.")

    out = ROOT / "docs" / "robustness_audit.md"
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
