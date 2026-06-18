"""Runtime loader for cached LTR predictions (offline-trained calibrator).

train_ltr.py (offline, sklearn) writes cache/ltr_scores.npz; this module only
loads it so rank.py stays sklearn-free at ranking time. Returns {candidate_id:
score} or None if the cache is absent.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LTR_CACHE = ROOT / "cache" / "ltr_scores.npz"


def load_ltr_scores():
    if not LTR_CACHE.exists():
        return None
    import numpy as np
    d = np.load(LTR_CACHE, allow_pickle=True)
    return {str(cid): float(s) for cid, s in zip(d["candidate_ids"], d["scores"])}
