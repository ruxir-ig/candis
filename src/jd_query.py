"""The query we embed the candidates against.

Critical design choice: we do NOT embed the raw job description. That JD is long
and roughly half of it is what they DON'T want ("if your AI experience is just
LangChain calling OpenAI...", "title-chasers", "consulting-only"). Embedding all
that pollutes the query vector with negative concepts.

Instead we embed a distilled, positive "ideal candidate" paragraph, phrased the
way a strong candidate would describe themselves — so it lands in the same
semantic neighbourhood as the candidate profile texts we compare it to.
"""

IDEAL_CANDIDATE_QUERY = (
    "Senior AI / machine learning engineer with five to nine years building "
    "production machine learning systems for search, retrieval, and ranking. "
    "Hands-on experience with embeddings-based retrieval using sentence-transformers, "
    "BGE, or E5 models, deployed to real users. Builds vector databases and hybrid "
    "search infrastructure with FAISS, Pinecone, Weaviate, Qdrant, Milvus, "
    "Elasticsearch, or OpenSearch, combining dense retrieval with BM25. Designs "
    "learning-to-rank and recommendation systems, and rigorous ranking evaluation "
    "with NDCG, MRR, MAP, and online A/B testing. Strong Python engineering. Has "
    "shipped end-to-end search, ranking, or recommendation systems at product "
    "companies, owning embedding drift, index refresh, and retrieval-quality "
    "regression in production."
)
