#!/usr/bin/env python3
"""Select buried-strong candidates for LLM expansion grading (pi's plan).

The top-200 LLM re-rank cache only sees the ensemble's top-200. Genuinely strong
candidates at ranks 201-800 never get LLM-graded. This selects a diverse,
evidence-rich set from that band so the judge can grade them — targeting "buried
fit" (high intrinsic quality dampened by availability), NOT "available but
mediocre".

Selection buckets (ranks 201-800), deduped, capped:
  50 high evidence_graph      cross-field corroboration
  40 high semantic            dense match to the ideal candidate
  40 high raw fit             strong before the availability multiplier
  30 high retrieval/vector    skills_score with vector-DB / retrieval skills
  30 Search/RecSys/NLP titles role-family match, buried by other factors
  20 high career/product      product-company ML pedigree

Excludes: candidates whose fit (pre-availability) is weak — they're only ranked
moderately because of behavioral availability, not buried talent.

Writes cache/llm_expansion_queue.txt (one candidate_id per line for llm_label.py).

    uv run python precompute/select_llm_expansion.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.ranker import score_all  # noqa: E402
from src.evidence_graph import evidence_graph_score, HIGH_STAKES_SKILLS  # noqa: E402

QUEUE_OUT = ROOT / "cache" / "llm_expansion_queue.txt"
MIN_FIT = 0.45  # below this pre-availability = not "buried fit", just weak

VECTOR_DB = {"faiss", "qdrant", "pinecone", "weaviate", "milvus",
             "opensearch", "elasticsearch", "vector"}
TITLE_FAMILY = ("search", "recommendation", "recsys", "nlp", "ranking",
                "retrieval", "relevance", "applied scientist")


def _has_vector_skill(candidate):
    names = {(s.get("name", "") or "").lower() for s in candidate.get("skills", []) or []}
    return any(v in n for n in names for v in VECTOR_DB)


def _title_family_match(candidate):
    t = (candidate.get("profile", {}).get("current_title", "") or "").lower()
    return any(f in t for f in TITLE_FAMILY)


def main():
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, ref = score_all(candidates)

    # Evidence graph score for the buried band.
    band = scored[200:800]  # ranks 201-800
    rows = []
    for r in band:
        c = r["candidate"]
        ev = evidence_graph_score(c)["score"]
        rows.append({
            "cid": r["candidate_id"],
            "candidate": c,
            "fit": r["fit"],
            "components": r["components"],
            "availability": r["availability"],
            "evidence": ev,
        })

    # Filter out "available but mediocre": keep only genuine buried fit.
    pool = [r for r in rows if r["fit"] >= MIN_FIT]
    print(f"Buried band ranks 201-800: {len(rows)}  with fit>={MIN_FIT}: {len(pool)}")

    def top_n(items, key, n):
        return [x["cid"] for x in sorted(items, key=lambda x: -key(x))[:n]]

    selected = []  # list of (cid, bucket)
    seen = set()

    def add_bucket(items, key, n, label):
        added = 0
        for x in sorted(items, key=lambda x: -key(x)):
            if x["cid"] in seen:
                continue
            seen.add(x["cid"])
            selected.append((x["cid"], label, key(x)))
            added += 1
            if added >= n:
                break
        return added

    counts = {}
    counts["evidence_graph"] = add_bucket(pool, lambda x: x["evidence"], 50, "high_evidence")
    counts["semantic"] = add_bucket(pool, lambda x: x["components"]["semantic"], 40, "high_semantic")
    counts["raw_fit"] = add_bucket(pool, lambda x: x["fit"], 40, "high_raw_fit")
    counts["retrieval_skills"] = add_bucket(
        [x for x in pool if _has_vector_skill(x["candidate"])],
        lambda x: x["components"]["skills"], 30, "retrieval_vector")
    counts["title_family"] = add_bucket(
        [x for x in pool if _title_family_match(x["candidate"])],
        lambda x: x["fit"], 30, "title_family")
    counts["career_product"] = add_bucket(pool, lambda x: x["components"]["career"], 20, "career_product")

    print("\nSelection by bucket (deduped):")
    for k, v in counts.items():
        print(f"  {k:<20} {v}")
    print(f"  TOTAL unique: {len(selected)}")

    # Write the queue
    QUEUE_OUT.parent.mkdir(exist_ok=True)
    with open(QUEUE_OUT, "w") as f:
        for cid, _, _ in selected:
            f.write(cid + "\n")

    # Preview
    cands_by_id = {c["candidate_id"]: c for c in candidates}
    rank_of = {r["cid"]: 201 + i for i, r in enumerate(rows)}
    print(f"\nWrote {QUEUE_OUT.relative_to(ROOT)} ({len(selected)} candidates)")
    print("\nPreview (first 15):")
    for cid, bucket, score in selected[:15]:
        p = cands_by_id[cid].get("profile", {})
        print(f"  #{rank_of.get(cid,'?'):<4} {cid}  [{bucket}]  "
              f"{p.get('current_title','')[:28]:28} @ {p.get('current_company','')[:16]}")


if __name__ == "__main__":
    main()
