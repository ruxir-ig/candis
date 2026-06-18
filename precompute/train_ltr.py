#!/usr/bin/env python3
"""Offline: train a Learning-to-Rank calibrator on LLM pseudo-labels.

The hand-tuned weight set in config.py is a judgment call. LTR asks: can a model
*learn* better feature weights from the LLM judge's continuous fit_score (0-10)?
We use the LLM re-rank grades (200) + hand golden labels (53) as weak supervision.

Two models, both offline-only:
  - HistGradientBoostingRegressor (the predictor; exported as cache/ltr_scores.npz)
  - Ridge linear (coefficients exported to cache/ltr_model.json for the deck —
    shows which features the judge rewards/penalises)

Runtime rank.py loads ONLY the cached predictions — no sklearn at ranking time.

Protocol: 5-fold CV reporting heldout NDCG, then refit on all labels and predict
for every post-filter candidate.

    uv run --with-optional ltr python precompute/train_ltr.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.loader import load_candidates  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.coarse_filter import passes_coarse_filter  # noqa: E402
from src.ranker import score_all, score_candidate, reference_date  # noqa: E402
from src.config import FIT_WEIGHTS_HYBRID  # noqa: E402
from src.semantic_matcher import semantic_scores  # noqa: E402
from src.scorers import (  # noqa: E402
    title_role_score, skills_score, career_score, experience_score,
    education_score, location_score, behavioral_availability,
)
from src.scorers.career_scorer import is_services_only, _is_services  # noqa: E402
from src.scorers.skills_scorer import SKILL_RELEVANCE  # noqa: E402
from src.evidence_graph import (  # noqa: E402
    evidence_graph_score, RETRIEVAL_CONCEPTS, HIGH_STAKES_SKILLS,
)

FEATURES = [
    "title", "skills", "career", "experience", "education", "location",
    "semantic", "availability", "evidence_graph",
    "product_company", "consulting_only",
    "retrieval_terms", "vector_db_skills",
    "assessment_support", "unsupported_ratio",
]

VECTOR_DB = {"faiss", "qdrant", "pinecone", "weaviate", "milvus",
             "opensearch", "elasticsearch", "vector"}


def _career_text(candidate):
    parts = []
    for r in candidate.get("career_history", []) or []:
        parts.append((r.get("description") or "") + " " + (r.get("title") or ""))
    s = candidate.get("profile", {}).get("summary") or ""
    return (s + " " + " ".join(parts)).lower()


def extract_features(candidate, ref, sem) -> dict:
    sig = candidate.get("redrob_signals", {}) or {}
    skills = candidate.get("skills", []) or []
    skill_names_l = [(s.get("name", "") or "").lower() for s in skills]
    career_text = _career_text(candidate)

    history = candidate.get("career_history", []) or []
    has_product = any(not _is_services(r.get("company", "")) for r in history)
    services_only = is_services_only(candidate)

    # retrieval term density in career prose
    retrieval_hits = sum(1 for t in RETRIEVAL_CONCEPTS if t in career_text)
    # vector-db skill count
    vdb = sum(1 for n in skill_names_l if any(v in n for v in VECTOR_DB))
    # assessment support
    assessments = sig.get("skill_assessment_scores", {}) or {}
    assess_vals = [v for v in assessments.values() if isinstance(v, (int, float))]
    assessment_support = (np.mean(assess_vals) / 100.0) if assess_vals else 0.0
    # unsupported expert skills (expert proficiency, ~0 duration)
    expert_skills = [s for s in skills if (s.get("proficiency") or "").lower() == "expert"]
    unsupported = sum(1 for s in expert_skills if not s.get("duration_months"))
    unsupported_ratio = unsupported / max(1, len(expert_skills)) if expert_skills else 0.0

    eg = evidence_graph_score(candidate)

    return {
        "title": title_role_score(candidate),
        "skills": skills_score(candidate),
        "career": career_score(candidate),
        "experience": experience_score(candidate),
        "education": education_score(candidate),
        "location": location_score(candidate),
        "semantic": sem.get(candidate["candidate_id"], 0.0) if sem else 0.0,
        "availability": behavioral_availability(candidate, ref),
        "evidence_graph": eg["score"],
        "product_company": 1.0 if has_product else (0.0 if services_only else 0.5),
        "consulting_only": 1.0 if services_only else 0.0,
        "retrieval_terms": min(1.0, retrieval_hits / 6.0),
        "vector_db_skills": min(1.0, vdb / 3.0),
        "assessment_support": assessment_support,
        "unsupported_ratio": unsupported_ratio,
    }


def load_labels():
    """{candidate_id: float target}. LLM fit_score/10 + hand grade/4, combined."""
    labels = {}
    llm_path = ROOT / "cache" / "llm_rerank.jsonl"
    if llm_path.exists():
        for line in open(llm_path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fs = d.get("fit_score")
            if fs is not None:
                labels[d["candidate_id"]] = float(fs) / 10.0  # normalize to [0,1]
    hand_path = ROOT / "eval" / "golden_labels.csv"
    if hand_path.exists():
        import csv
        for r in csv.DictReader(open(hand_path)):
            g = (r.get("grade") or "").strip()
            if g:
                labels[r["candidate_id"]] = int(float(g)) / 4.0
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    t0 = time.time()
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    scored, ref = score_all(candidates)
    sem = semantic_scores()
    print(f"Pool: {len(scored):,}. Extracting features...")

    # Feature matrix for the whole post-filter pool.
    feats_by_id = {}
    X_ids, X = [], []
    for r in scored:
        f = extract_features(r["candidate"], ref, sem)
        feats_by_id[r["candidate_id"]] = f
        X_ids.append(r["candidate_id"])
        X.append([f[k] for k in FEATURES])
    X = np.array(X, dtype=np.float64)
    print(f"Features extracted for {len(X):,} candidates in {time.time()-t0:.1f}s")

    labels = load_labels()
    # Align labeled subset
    lab_idx = [i for i, cid in enumerate(X_ids) if cid in labels]
    lab_y = np.array([labels[X_ids[i]] for i in lab_idx])
    lab_X = X[lab_idx]
    print(f"Labels: {len(lab_y)} (llm={0} + hand={0} combined) — "
          f"target distribution: min={lab_y.min():.2f} mean={lab_y.mean():.2f} max={lab_y.max():.2f}")

    # ---- 5-fold CV: heldout NDCG@10 ----------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_ndcg10, cv_ndcg50 = [], []

    def ndcg_at(order_targets, k):
        gains = (2 ** np.array(order_targets) - 1)
        dcg = np.sum(gains[:k] / np.log2(np.arange(2, k + 2)))
        ideal = np.sort(gains)[::-1][:k]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        return dcg / idcg if idcg > 0 else 0.0

    for fold, (tr, va) in enumerate(kf.split(lab_X)):
        m = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.08, max_iter=200,
            l2_regularization=0.5, random_state=args.seed).fit(lab_X[tr], lab_y[tr])
        pred = m.predict(lab_X[va])
        order = np.argsort(-pred)
        cv_ndcg10.append(ndcg_at(lab_y[va][order], 10))
        cv_ndcg50.append(ndcg_at(lab_y[va][order], min(50, len(va))))
    print(f"\nCV heldout NDCG@10 = {np.mean(cv_ndcg10):.4f} ± {np.std(cv_ndcg10):.4f}")
    print(f"CV heldout NDCG@50 = {np.mean(cv_ndcg50):.4f} ± {np.std(cv_ndcg50):.4f}")

    # Baseline: does the hand-tuned weighted score order the heldout as well?
    # (use the ensemble score restricted to labeled set, per-fold)
    base_score = {X_ids[i]: sum(FIT_WEIGHTS_HYBRID.get(k, 0) * feats_by_id[X_ids[i]][k]
                                for k in FIT_WEIGHTS_HYBRID) for i in lab_idx}
    cv_base_ndcg10 = []
    for fold, (tr, va) in enumerate(kf.split(lab_X)):
        va_ids = [X_ids[lab_idx[i]] for i in va]
        va_y = lab_y[va]
        order = sorted(range(len(va_ids)), key=lambda j: -base_score[va_ids[j]])
        cv_base_ndcg10.append(ndcg_at(va_y[order], 10))
    print(f"Baseline (hand-tuned) CV NDCG@10 = {np.mean(cv_base_ndcg10):.4f} ± {np.std(cv_base_ndcg10):.4f}")

    # ---- Refit on all labels, predict full pool ----------------------------
    final = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.08, max_iter=200,
        l2_regularization=0.5, random_state=args.seed).fit(lab_X, lab_y)
    preds = final.predict(X)
    # Normalize to [0,1]
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-9)

    out = ROOT / "cache" / "ltr_scores.npz"
    np.savez(out, candidate_ids=np.array(X_ids), scores=preds.astype(np.float32))
    print(f"\nWrote {out.relative_to(ROOT)} — {len(preds):,} predictions")

    # ---- Linear model for interpretability (deck) --------------------------
    ridge = Ridge(alpha=1.0).fit(lab_X, lab_y)
    coefs = {k: float(c) for k, c in zip(FEATURES, ridge.coef_)}
    model_json = {
        "model": "Ridge (interpretability only; HGBR is the predictor)",
        "features": FEATURES,
        "coefficients": coefs,
        "intercept": float(ridge.intercept_),
        "n_labels": len(lab_y),
        "cv_ndcg10_hgbr": float(np.mean(cv_ndcg10)),
        "cv_ndcg10_baseline": float(np.mean(cv_base_ndcg10)),
    }
    (ROOT / "cache" / "ltr_model.json").write_text(json.dumps(model_json, indent=2))
    print("Ridge coefficients (deck — what the judge rewards/penalises):")
    for k, v in sorted(coefs.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:+.3f}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
