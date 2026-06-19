# Research Looping Plan — India Runs / Redrob Candidate Ranking

**Current loop number:** `2`
**Current date:** 19 June 2026  
**Submission deadline:** 2 July 2026, 23:59 IST  
**Project status:** Final submission is currently frozen, validated, audited, and ready. Future work must happen in controlled research loops and must not destabilize the current final output.

---

## 0. Purpose of this document

This file defines the operating procedure for the remaining research cycles before submission.

The goal is to keep improving the project with research-backed ideas **without breaking the already-strong final system**.

The current final system already includes:

- hard honeypot/non-technical filters
- hybrid rule + semantic ranking
- behavioral availability multiplier
- cached top-200 LLM rerank
- evidence-guided LLM expansion with top-20 frozen
- manual top-100 audit
- robustness audit
- deck and final output artifacts

Therefore, every future loop should be treated as a **sandbox experiment**. A loop can improve the final ranking only if it passes strict adoption gates.

---

## 1. Non-negotiable final-system rule

The current final submission is the safety baseline.

Do **not** modify the default final ranking unless the new method clearly passes all adoption gates in Section 7.

If a method is interesting but does not clearly improve the final output, it should be documented as an ablation and left out of the default final ranker.

Preferred mindset:

> Build a ranking laboratory. Keep only methods that improve recruiter trust under constraints.

---

## 2. Loop file protocol

For each loop, follow a dedicated loop file:

```text
loop_<loop_number>.md
```

Examples:

```text
loop_1.md
loop_2.md
loop_3.md
```

At the beginning of work, the agent must:

1. Read `looping_plan.md`.
2. Check `Current loop number` above.
3. Look for the corresponding loop file.

For current loop `1`, the agent should look for:

```text
loop_1.md
```

If the file does not exist, the agent must create it before doing implementation work.

The loop file should contain:

```md
# Loop <N> Plan

Date started:
Current time:
Submission deadline:
Goal of this loop:

## Research papers / ideas considered

## Candidate approaches

## Feasibility analysis

## Expected speed of progress

## Implementation plan

## Evaluation plan

## Adoption gates

## Results

## Final decision
```

The loop file is the working scratchpad and final report for that loop.

---

## 3. How each new loop should be planned

Each new loop should begin by finding or reviewing research papers/ideas related to the problem statement:

> Build an AI system that ranks candidates the way a great recruiter would — understanding role fit, career evidence, skills, behavioral signals, platform activity, and anti-keyword-stuffing robustness.

The agent should look for papers or technical ideas that are **not already implemented** in the final system.

Already implemented or tested:

- BM25 + Reciprocal Rank Fusion — tested and rejected
- Evidence graph ranker — tested as ranker, rejected; retained for reasoning
- Robustness audit against keyword stuffing — completed
- Learning-to-rank with pseudo-labels — tested and rejected
- GNN / spectral graph embedding candidate matching — tested and rejected
- Cached LLM top-200 reranking — adopted
- Evidence-guided LLM expansion — adopted
- Multi-agent panel — considered/deferred

Future loop ideas may include, but are not limited to:

- pairwise LLM preference ranking for local candidate swaps
- role-aware expert mixtures
- adversarial resume / prompt-injection robustness
- structured JD decomposition / query understanding
- uncertainty-aware ranking or confidence intervals
- fairness/bias audit
- better calibration of score outputs
- local listwise reranking around ranks 21-100
- counterfactual candidate perturbation tests
- rank stability analysis across missing caches / ablations

For every paper/idea, explain in simple words:

1. What the idea is.
2. Why it might help this candidate-ranking challenge.
3. What part of the current system it would affect.
4. How hard it is to implement.
5. How risky it is to the final ranking.
6. How quickly progress can be made.
7. Whether it should be code, audit-only, deck-only, or rejected.

---

## 4. Required check-ins with pi-agent / advisor

When asking pi-agent, an advisor, or another agent for an opinion, doubt, check-in, or new loop plan, always include:

- current date
- current local time
- current loop number
- branch name
- what changed since last check-in
- current metrics, if relevant
- whether final submission files changed
- exact question being asked

Template:

```md
Date/time: 19 June 2026, HH:MM IST
Loop: 1
Branch: research/<name>

What changed:
- ...

Current validation:
- rank.py runtime:
- submission validity:
- hand qrel:
- top-20 changed? yes/no
- top-100 changed? yes/no

Question:
- ...
```

Important note: loops will **not** run 24/7. Each loop should be resumable from its `loop_<N>.md` file.

---

## 5. Branching protocol

Never experiment directly on the frozen final state without a branch.

For each loop:

```bash
git checkout main
git status --short
git checkout -b research/loop-<N>-<short-topic>
```

Examples:

```bash
git checkout -b research/loop-1-pairwise-llm
git checkout -b research/loop-2-expert-mixture
git checkout -b research/loop-3-adversarial-audit
```

At the end of the loop:

- If the method passes adoption gates, merge to main.
- If not, keep the branch or document results and do not merge ranking changes.
- Documentation-only results may be merged if useful and clean.

---

## 6. Runtime and dependency rules

The challenge ranking step must remain:

- <= 5 minutes
- <= 16 GB RAM
- CPU-only
- no network

Runtime `rank.py` should remain lightweight.

Allowed at runtime:

- Python stdlib
- numpy
- loading cached `.jsonl`, `.json`, `.npz`, or small model artifacts
- deterministic scoring / sorting

Avoid at runtime:

- torch
- sklearn
- scipy
- networkx
- sentence-transformers
- LLM/API calls
- web/network requests
- model training

Heavy research work belongs in `precompute/` or `eval/`, not in the final ranking path.

Use `uv` moving forward.

Common commands:

```bash
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv
uv run python validate_submission.py output/submission.csv
uv run python eval/evaluate.py --qrel hand
```

If adding dependencies, prefer optional/dev groups where possible. Keep the final runtime path minimal.

---

## 7. Adoption gates for any new method

A new method can affect the default final ranking only if all of these pass:

1. **Top-20 protection:** top-20 remains unchanged, unless there is an extremely strong manually approved reason.
2. **Metric safety:** no drop in hand NDCG@10.
3. **NDCG@50 safety:** no material drop in hand NDCG@50.
4. **Top-100 hygiene:** 0 honeypots, 0 non-technical titles, 0 services-only profiles, 0 keyword-stuffer suspects.
5. **Manual audit:** every changed top-100 candidate is manually audited.
6. **Small blast radius:** ideally <= 10 top-100 changes.
7. **Runtime safety:** final ranking remains <= 5 min, CPU-only, no network.
8. **Reproducibility:** method uses committed or clearly documented cached artifacts.
9. **Explainability:** the change has a clear recruiter-trust explanation.
10. **Rollback:** `--no-<feature>` or equivalent fallback exists if the change becomes default.

If any of these fail, document the method as an ablation and do not make it default.

---

## 8. Evaluation checklist for every loop

At minimum, run:

```bash
uv run python rank.py --candidates data/candidates.jsonl --out output/submission.csv
uv run python validate_submission.py output/submission.csv
uv run python eval/evaluate.py --qrel hand
```

If the loop affects ranking, also compare:

- old top-100 vs new top-100
- entrants and leavers
- top-20 diff
- top-50 diff
- top-100 hygiene
- runtime
- graceful degradation if relevant

Recommended loop artifacts:

```text
docs/ablation_<method>.md
docs/audit_<method>.md
eval/<method>_audit.py
precompute/<method>.py
cache/<method>.jsonl or cache/<method>.npz
```

---

## 9. Suggested Loop 1 topic

Suggested starting topic for `loop_1.md`:

> Pairwise LLM preference audit for expansion entrants vs displaced candidates.

Why this is high value:

- It validates the latest adopted evidence-guided expansion.
- It is local and low-risk.
- It does not require new architecture.
- It can provide a strong audit result such as: “pairwise recruiter judge preferred promoted candidates over displaced candidates in X/Y comparisons.”

Possible implementation:

```text
precompute/llm_pairwise.py
cache/llm_pairwise.jsonl
eval/audit_pairwise_expansion.py
docs/pairwise_expansion_audit.md
```

Safety rule:

- Use pairwise audit primarily for validation.
- Do not reorder top-20.
- Do not change final ranking unless pairwise results expose a clear issue.

---

## 10. GitHub / artifact upload guidance

It is now high time to upload the project to GitHub.

Before uploading:

1. Confirm no secrets/API keys are present.
2. Confirm raw challenge data policy is respected.
3. Confirm ignored generated artifacts are handled intentionally.
4. Confirm README explains how to regenerate final outputs.
5. Confirm `cache/embeddings.npz` handling is clear.

Current caveat:

- `cache/embeddings.npz` is 136 MB and likely too large for normal GitHub without Git LFS.
- It is needed for exact hybrid reproduction but can be regenerated via the embedding setup.
- For the challenge submission, include it alongside the repo or document regeneration clearly.

Generated artifacts:

```text
output/submission.csv
output/deck.pdf
output/deck.pptx
```

These may remain ignored in git if the challenge accepts them as separate uploads. If the GitHub repo itself must contain them, update `.gitignore` intentionally and commit only the required artifacts.

Markdown files:

- Commit important methodology/audit markdown files if they help reviewers understand the system.
- Avoid committing noisy scratch notes unless they are useful.
- `loop_<N>.md` files may be committed if they document meaningful research decisions.

PDF/PPTX files:

- Commit only if required by challenge instructions or useful for GitHub reviewers.
- Otherwise upload them separately as official deliverables.

Recommended GitHub readiness check:

```bash
git status --short
git ls-files | sort
rg -n "API_KEY|NVIDIA|SECRET|TOKEN|PASSWORD|Bearer" . --glob '!data/candidates.jsonl' --glob '!cache/*.npz'
```

---

## 11. Loop completion protocol

At the end of each loop, update the loop file with:

```md
## Results

## Metrics

## Entrants/leavers if ranking changed

## Manual audit summary

## Decision
Adopted / Rejected / Documentation-only / Deferred

## Next recommended loop
```

Then update this file if moving to a new loop:

```md
Current loop number: <N+1>
```

Only increment the loop number after the previous loop is complete or explicitly abandoned.

---

## 12. Stop condition

Stop experimenting and submit/freeze if any of the following are true:

- There are <= 3 days left.
- A new method causes confusing instability.
- The final deck/repo packaging is not ready.
- The final output has not been validated recently.
- A change improves internal metrics but fails manual recruiter-trust audit.

The final project should prioritize trust, reproducibility, and clean delivery over last-minute novelty.
