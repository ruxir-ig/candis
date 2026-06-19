#!/usr/bin/env python3
"""Offline: build the heterogeneous candidate graph for node2vec / GNN.

Nodes:
  - candidate   (CAND_xxx)
  - skill       (normalized skill name)
  - concept     (retrieval/search concept — the job's required vocabulary)
  - company     (normalized company name)

Edges (undirected, weighted):
  candidate --skill--> skill      weight = relevance (from skills_scorer)
  candidate --concept--> concept  weight = 1 if career text mentions the concept
  candidate --company--> company  weight = 1

The job node's required concepts are the seed; a candidate structurally close to
them in the graph (shared skills/companies with known-strong engineers) embeds
nearby. node2vec over this graph produces the candidate embedding train_gnn.py
regresses on.

    uv run --extra graph python precompute/build_graph.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402

from src.loader import load_candidates  # noqa: E402
from src.honeypot_detector import is_honeypot  # noqa: E402
from src.coarse_filter import passes_coarse_filter  # noqa: E402
from src.scorers.skills_scorer import SKILL_RELEVANCE  # noqa: E402
from src.evidence_graph import RETRIEVAL_CONCEPTS, MATURITY_TERMS  # noqa: E402

CACHE = ROOT / "cache"


def _skill_relevance(name: str) -> float:
    n = (name or "").lower()
    best = 0.0
    for kw, w in SKILL_RELEVANCE.items():
        if kw in n and w > best:
            best = w
    return best


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _node_id(prefix: str, name: str) -> str:
    """Edgelist-safe node id: prefix + sanitized name (no spaces)."""
    safe = "".join(c if c.isalnum() else "_" for c in (name or "").strip().lower())
    return f"{prefix}::{safe}"


def main():
    t0 = time.time()
    candidates = list(load_candidates(ROOT / "data" / "candidates.jsonl"))
    kept = [c for c in candidates if not is_honeypot(c) and passes_coarse_filter(c)]
    print(f"Building graph over {len(kept):,} post-filter candidates...")

    G = nx.Graph()
    concepts = RETRIEVAL_CONCEPTS | MATURITY_TERMS

    for c in kept:
        cid = c["candidate_id"]
        G.add_node(cid, type="candidate")

        # candidate --skill--> skill
        for s in c.get("skills", []) or []:
            sn = _norm(s.get("name", ""))
            if not sn:
                continue
            rel = _skill_relevance(sn)
            if rel <= 0:
                continue
            nid = _node_id("skill", sn)
            G.add_node(nid, type="skill")
            G.add_edge(cid, nid, weight=rel)

        # candidate --concept--> concept (career text corroboration)
        career = " ".join((r.get("description") or "") + " " + (r.get("title") or "")
                          for r in c.get("career_history", []) or []).lower()
        summary = (c.get("profile", {}).get("summary") or "").lower()
        text = summary + " " + career
        for con in concepts:
            if con in text:
                nid = _node_id("concept", con)
                G.add_node(nid, type="concept")
                G.add_edge(cid, nid, weight=1.0)

        # candidate --company--> company
        for r in c.get("career_history", []) or []:
            co = _norm(r.get("company", ""))
            if co:
                nid = _node_id("company", co)
                G.add_node(nid, type="company")
                G.add_edge(cid, nid, weight=0.5)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    by_type = {}
    for n, d in G.nodes(data=True):
        by_type[d.get("type", "?")] = by_type.get(d.get("type", "?"), 0) + 1
    print(f"Graph: {n_nodes:,} nodes ({by_type}), {n_edges:,} edges in {time.time()-t0:.1f}s")

    CACHE.mkdir(exist_ok=True)
    # Save as weighted edge list (compact) + candidate node list.
    nx.write_weighted_edgelist(G, CACHE / "graph.edgelist")
    (CACHE / "graph_nodes.json").write_text(__import__("json").dumps({
        "candidates": [n for n, d in G.nodes(data=True) if d.get("type") == "candidate"],
        "n_nodes": n_nodes, "n_edges": n_edges, "by_type": by_type,
    }))
    print(f"Wrote cache/graph.edgelist + graph_nodes.json "
          f"({CACHE.stat() if False else ''}{time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
