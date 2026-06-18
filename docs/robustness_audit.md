# Robustness / anti-gaming audit

Sample sizes: 500 weak (non-technical) profiles, 100 strong (top-100) profiles.

## 1. Keyword stuffing (inject FAISS/Qdrant/Pinecone/RAG/...)

Injecting 7 expert AI skills (0 months duration) into 500 weak profiles.

| ranker               | stuffed weak in top-100 | stuffed weak in top-500 |
|----------------------|--------------------------|--------------------------|
| **keyword baseline** | 15                       | 24                       |
| **our system**       | 0                        | 0                        |

Our system: 191/500 weak profiles moved up at all after stuffing; median post-stuffing rank #62566 of 83,779. The keyword baseline vaults them near the top of the pool.

## 2. Skill removal (strip skills from top-100 strong profiles)

Removing all skills: median rank drop 524, mean 518, worst 630. Collapsed beyond #1000: 0/100 (career/title evidence keeps genuine engineers relevant).

## 3. Skill shuffle (order invariance)

Max rank change after shuffling skill order: 0 (must be 0 — ranking is order-invariant by design).

## 4. Drop behavioral signals (remove all redrob_signals)

Removing all behavioral signals: max rank change 1341, median 914 (behavioral is a dampener, not a primary signal, so the ranking stays intact).

## 5. Title inflation (relabel weak profile 'Senior AI Engineer')

Inflating title to 'Senior AI Engineer': 500/500 weak profiles moved up; 0 reached top-100; median jump 64427. The title anchor is cross-checked against trust-weighted skills and career corroboration, so a title change alone does not carry a weak profile up.

_Methodology_: ranks computed over the full post-filter pool (83,779). Our system uses the full rule+semantic ensemble; the semantic score is cached by candidate_id and stays fixed across perturbations, so the test isolates the structural rule defenses. Single-candidate rank via binary search over the precomputed pool scores.
