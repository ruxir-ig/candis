# Ablation: Learning-to-Rank calibrator

**Status: ABSTAINED from ranking** (overfits the small label set); **Ridge
coefficient analysis retained** as independent validation of the hand-tuned
weights.

## What we built

An offline LTR calibrator (`precompute/train_ltr.py`, sklearn, `--extra ltr`):
- **Features (15):** the 6 rule components + semantic + availability +
  evidence_graph + product_company + consulting_only + retrieval_terms +
  vector_db_skills + assessment_support + unsupported_ratio.
- **Labels (219):** LLM re-rank fit_score/10 (200, from top-200) + hand golden
  grades/4 (53). Combined target in [0,1], distribution heavily skewed strong
  (mean 0.79 — the LLM grades are all top-200 candidates).
- **Models:** HistGradientBoostingRegressor (predictor → `cache/ltr_scores.npz`)
  + Ridge (coefficients → `cache/ltr_model.json` for the deck).
- Runtime (`src/ltr_model.py`) loads only the cached predictions — sklearn-free.

## Result: overfits, does NOT beat hand-tuned weights

5-fold CV heldout NDCG@10:

| model          | heldout NDCG@10 |
|----------------|-----------------|
| hand-tuned ens | **0.9426**      |
| HGBR (LTR)     | 0.9003          |

The LTR model underperforms the hand-tuned weights. Root cause (the plan's
predicted "Failure 1: overfits small labels"):
- only 219 labels, and they're **distribution-skewed** — 200 come from the
  top-200 (nearly all grade-4), so the model rarely sees a true weak candidate
  and can't learn the decision boundary well.
- gradient boosting on 219 rows overfits the folds.

## Beware the circularity trap

Blending LTR into the ranking *appears* to push the **hand qrel composite to
1.0000** — but that is **label leakage**: the hand golden labels are in the
training set. The LLM qrel (more honest) shows LTR *hurts* (composite 0.9765 →
0.9553, ndcg@10 1.0 → 0.947) — the same saturation problem (active reordering
disrupts the perfect top-10). The 5-fold CV is the only trustworthy number, and
it says LTR loses.

## The real deliverable: Ridge coefficients validate the design

The linear model's learned weights independently confirm the hand-tuned feature
choices:

| feature           | Ridge coef |
|-------------------|------------|
| skills            | **+0.394** |
| evidence_graph    | **+0.365** |
| title             | **+0.229** |
| experience        | +0.175     |
| retrieval_terms   | +0.166     |
| location          | +0.141     |
| education         | +0.130     |
| semantic          | +0.078     |
| availability      | +0.074     |
| career            | -0.008     |
| assessment_support| -0.015     |
| product_company   | -0.056     |

Skills, cross-field evidence, and title are the three dominant features —
exactly what the hand-tuned `FIT_WEIGHTS_HYBRID` already prioritises (title 0.28,
skills 0.23, +semantic 0.15). The judge de-emphasises raw career/product signals
relative to our weights, suggesting a small tuning opportunity, but not enough to
justify replacing interpretable weights with an overfit black box on 219 labels.

## Decision

LTR is **not used in the final ranking**. The Ridge analysis is kept for the
deck as independent validation. A future improvement would require a much larger,
de-saturated label set (e.g. LLM-grading a stratified sample across the full fit
range) before LTR can generalise — noted as future work.

## Deck framing
> "We trained a gradient-boosting LTR calibrator on 219 LLM + hand labels. 5-fold
> CV showed it overfit the small, distribution-skewed set (heldout NDCG@10 0.90
> vs our hand-tuned 0.94) — the labels are concentrated in the top-200, so the
> model rarely sees a weak candidate. A Ridge linear fit, however, independently
> confirmed that skills, title, and cross-field evidence are the dominant
> features, matching our hand-tuned weights. We kept LTR as an ablation and
> retained the interpretable weights, noting that a larger stratified label set
> is needed before learned weights can generalise."
