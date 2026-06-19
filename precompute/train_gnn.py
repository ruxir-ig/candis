#!/usr/bin/env python3
"""Offline: graph embedding + regressor (the GNN ranker).

We tried node2vec but its pure-Python alias-sampling didn't scale to 84K weighted
nodes within our compute budget. Pivoted to a spectral embedding: TruncatedSVD
on the sparse candidate-attribute adjacency matrix — a scalable graph
factorization that places candidates with shared skills/companies/concepts near
each other in a low-dimensional space. Then a regressor learns fit from the
embedding (the GNN's learned signal).

5-seed stability check (the plan's GNN acceptance gate): if the regressor's
ranking varies wildly across seeds, the graph signal isn't stable enough.

    uv run --extra graph python precompute/train_gnn.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

CACHE = ROOT / "cache"


def load_labels():
    labels = {}
    for line in open(CACHE / "llm_rerank.jsonl"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        fs = d.get("fit_score")
        if fs is not None:
            labels[d["candidate_id"]] = float(fs) / 10.0
    import csv
    for r in csv.DictReader(open(ROOT / "eval" / "golden_labels.csv")):
        g = (r.get("grade") or "").strip()
        if g:
            labels[r["candidate_id"]] = int(float(g)) / 4.0
    return labels


def ndcg_at(order_targets, k):
    gains = (2 ** np.array(order_targets) - 1)
    dcg = np.sum(gains[:k] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(gains)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    t0 = time.time()
    # Load the weighted edgelist into a sparse adjacency matrix.
    rows, cols, data = [], [], []
    node_index = {}
    with open(CACHE / "graph.edgelist") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            a, b, w = parts[0], parts[1], float(parts[2])
            for n in (a, b):
                if n not in node_index:
                    node_index[n] = len(node_index)
            rows.append(node_index[a])
            cols.append(node_index[b])
            data.append(w)
    n = len(node_index)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A + A.T  # symmetrize (undirected)
    idx_of = {v: k for k, v in node_index.items()}
    print(f"Loaded graph: {n:,} nodes, {A.nnz:,} directed entries in {time.time()-t0:.1f}s")

    cand_ids = [idx_of[i] for i in range(n) if idx_of[i].startswith("CAND_")]
    cand_rows = [node_index[c] for c in cand_ids]
    print(f"Candidate nodes: {len(cand_ids):,}")

    # Spectral embedding: TruncatedSVD on the candidate x attribute adjacency.
    ts = time.time()
    svd = TruncatedSVD(n_components=args.dim, random_state=0)
    # A is node x node; restrict rows to candidates for the embedding.
    A_cand = A[cand_rows, :]
    emb_all = svd.fit_transform(A_cand)  # (n_cand, dim)
    print(f"Spectral embedding ({args.dim}-d) in {time.time()-ts:.1f}s "
          f"(explained var: {svd.explained_variance_ratio_.sum():.2%})")

    emb = {cid: emb_all[i] for i, cid in enumerate(cand_ids)}

    labels = load_labels()
    lab_ids = [cid for cid in cand_ids if cid in labels]
    print(f"Labels usable: {len(lab_ids)}")

    X = np.array([emb[cid] for cid in lab_ids])
    y = np.array([labels[cid] for cid in lab_ids])
    Xall = np.array([emb[cid] for cid in cand_ids])

    # ---- per-seed CV stability ----
    seed_list = list(range(1, args.seeds + 1))
    seed_ndcg, seed_preds = [], []
    for seed in seed_list:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        folds = []
        for tr, va in kf.split(X):
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1,
                                              max_iter=150, l2_regularization=1.0,
                                              random_state=seed).fit(X[tr], y[tr])
            order = np.argsort(-m.predict(X[va]))
            folds.append(ndcg_at(y[va][order], 10))
        seed_ndcg.append(np.mean(folds))

        mfull = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1,
                                              max_iter=150, l2_regularization=1.0,
                                              random_state=seed).fit(X, y)
        pred = mfull.predict(Xall)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
        seed_preds.append({cid: float(p) for cid, p in zip(cand_ids, pred)})

    print(f"\n5-fold CV NDCG@10 by seed (graph-embedding only):")
    for s, ncg in zip(seed_list, seed_ndcg):
        print(f"  seed {s}: {ncg:.4f}")
    print(f"  mean={np.mean(seed_ndcg):.4f} std={np.std(seed_ndcg):.4f}")

    # Baseline: hand-tuned ensemble restricted to labeled set
    from src.ranker import score_all
    from src.loader import load_candidates
    scored, _ = score_all(list(load_candidates(ROOT / "data" / "candidates.jsonl")))
    base_rank = {r["candidate_id"]: i for i, r in enumerate(scored)}
    base_order = sorted(lab_ids, key=lambda c: base_rank.get(c, 10**9))
    print(f"Baseline (hand-tuned ensemble) NDCG@10 on labeled set: "
          f"{ndcg_at([labels[c] for c in base_order], 10):.4f}")

    # Stability: top-100 overlap across seeds
    top100s = [set(sorted(p, key=lambda c: -p[c])[:100]) for p in seed_preds]
    print(f"\nTop-100 overlap across seeds:")
    min_ov = 100
    for i in range(len(seed_preds)):
        for j in range(i + 1, len(seed_preds)):
            ov = len(top100s[i] & top100s[j])
            min_ov = min(min_ov, ov)
            print(f"  seed {seed_list[i]} vs {seed_list[j]}: {ov}/100")

    # Average predictions across seeds
    avg = {cid: float(np.mean([sp[cid] for sp in seed_preds])) for cid in cand_ids}
    out = CACHE / "gnn_scores.npz"
    np.savez(out, candidate_ids=np.array(cand_ids),
             scores=np.array([avg[c] for c in cand_ids], dtype=np.float32))
    print(f"\nWrote {out.relative_to(ROOT)} (seed-averaged, {len(avg):,} cands)")

    summary = {
        "model": f"spectral embedding (TruncatedSVD, dim={args.dim}) + HistGBR",
        "note": "node2vec didn't scale to 84K weighted nodes; pivoted to spectral.",
        "n_labels": len(lab_ids),
        "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        "cv_ndcg10_by_seed": dict(zip(seed_list, [round(x, 4) for x in seed_ndcg])),
        "cv_ndcg10_mean": round(float(np.mean(seed_ndcg)), 4),
        "cv_ndcg10_std": round(float(np.std(seed_ndcg)), 4),
        "top100_overlap_min": min_ov,
    }
    (CACHE / "gnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
