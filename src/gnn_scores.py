"""Runtime loader for cached GNN/graph scores. Loads cache/gnn_scores.npz only;
no sklearn/scipy at ranking time."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GNN_CACHE = ROOT / "cache" / "gnn_scores.npz"


def load_gnn_scores():
    if not GNN_CACHE.exists():
        return None
    import numpy as np
    d = np.load(GNN_CACHE, allow_pickle=True)
    return {str(cid): float(s) for cid, s in zip(d["candidate_ids"], d["scores"])}
