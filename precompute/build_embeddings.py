"""OFFLINE precompute: embed the JD query + every candidate profile, cache to disk.

This is the ONLY step that needs torch / sentence-transformers, and it runs
outside the challenge's 5-minute budget. Run it with the embedding venv:

    .venv-embed/bin/python precompute/build_embeddings.py

Output: cache/embeddings.npz  (candidate_ids, embeddings [N x D], jd_embedding [D])
The ranking step (rank.py) only ever loads this .npz with numpy — no torch.

Why this split: a system that calls a model per candidate can't scale to a 200K
pool in 5 minutes on CPU. Precomputing once and reusing cached vectors is exactly
the latency/quality tradeoff the JD says it's testing for.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_candidates  # noqa: E402
from src.profile_text import candidate_text  # noqa: E402
from src.jd_query import IDEAL_CANDIDATE_QUERY  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
OUT = ROOT / "cache/embeddings.npz"


def main():
    t0 = time.time()
    print(f"Loading candidates ...")
    candidates = list(load_candidates(ROOT / "data/candidates.jsonl"))
    ids = [c["candidate_id"] for c in candidates]
    texts = [candidate_text(c) for c in candidates]
    print(f"  {len(texts):,} profile texts built in {time.time()-t0:.1f}s")

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # normalize_embeddings=True -> cosine similarity is a plain dot product later.
    print("Encoding profiles (this is the slow part) ...")
    emb = model.encode(
        texts, batch_size=256, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)

    jd_emb = model.encode(
        [IDEAL_CANDIDATE_QUERY], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        candidate_ids=np.array(ids),
        embeddings=emb,
        jd_embedding=jd_emb,
        model_name=np.array(MODEL_NAME),
    )
    print(f"Saved {emb.shape} embeddings to {OUT} in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
