"""
retrieval.py – Contextual Retrieval pipeline.

Takes a query string, embeds it, searches both semantic (Qdrant vectors)
and lexical (BM25) indexes, fuses the results, reranks with Cohere,
and returns the top-K chunks.
"""

import os
import re

from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient

from localGeneration import generate, start_server, stop_server

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-english-v3.0"
SEMANTIC_COLLECTION = "contextualized_chunks"
BM25_COLLECTION = "bm25_chunks"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.3


# ── Step 1: Embed the query ─────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Embed the query using Cohere with input_type='search_query'."""
    co = cohere.ClientV2(api_key=API_KEY)
    response = co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]


# ── Step 2: Semantic search ──────────────────────────────────────────

def semantic_search(query_embedding: list[float], top_n: int = 20) -> list[dict]:
    """Search Qdrant for the closest vectors. Returns [{id, text, score}]."""
    client = QdrantClient(path="./qdrant_db")
    results = client.query_points(
        collection_name=SEMANTIC_COLLECTION,
        query=query_embedding,
        limit=top_n,
    )
    hits = []
    for point in results.points:
        hits.append({
            "id": point.id,
            "text": point.payload["text"],
            "score": point.score,
        })
    client.close()
    return hits


# ── Step 3: BM25 search ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(query: str, top_n: int = 20) -> list[dict]:
    """Score all chunks with BM25 and return top results."""
    client = QdrantClient(path="./qdrant_db")

    # Load global BM25 metadata (stored at id=0)
    meta = client.retrieve(BM25_COLLECTION, ids=[0])[0].payload
    k1 = meta["k1"]
    b = meta["b"]
    avgdl = meta["avgdl"]
    idf = meta["idf"]

    # Load all chunk points (ids 1..doc_count)
    doc_ids = list(range(1, meta["doc_count"] + 1))
    points = client.retrieve(BM25_COLLECTION, ids=doc_ids)
    client.close()

    query_terms = tokenize(query)
    scores = []

    for point in points:
        tf_map = point.payload["tf_map"]
        doc_length = point.payload["doc_length"]
        score = 0.0

        for term in query_terms:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            term_idf = idf.get(term, 0.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / avgdl)
            score += term_idf * numerator / denominator

        scores.append({
            "id": point.id,
            "text": point.payload["text"],
            "score": score,
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


# ── Step 4: Rank fusion ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Combine semantic and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across both result lists.
    k=60 is the standard constant that prevents top ranks from dominating.
    """
    fused = {}

    for rank, hit in enumerate(semantic_results, start=1):
        chunk_id = hit["id"]
        if chunk_id not in fused:
            fused[chunk_id] = {"id": chunk_id, "text": hit["text"], "rrf_score": 0.0}
        fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    for rank, hit in enumerate(bm25_results, start=1):
        chunk_id = hit["id"]
        if chunk_id not in fused:
            fused[chunk_id] = {"id": chunk_id, "text": hit["text"], "rrf_score": 0.0}
        fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return results


# ── Step 5: Rerank ───────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_n: int = TOP_K) -> list[dict]:
    """Rerank candidates using Cohere's rerank model."""
    co = cohere.ClientV2(api_key=API_KEY)
    docs = [c["text"] for c in candidates]
    response = co.rerank(
        query=query,
        documents=docs,
        model=RERANK_MODEL,
        top_n=top_n,
    )
    reranked = []
    for result in response.results:
        candidate = candidates[result.index]
        reranked.append({
            "id": candidate["id"],
            "text": candidate["text"],
            "relevance_score": result.relevance_score,
        })
    return reranked


# ── Step 6: Generate answer ──────────────────────────────────────────

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate an answer using the local LM Studio model."""
    context = "\n\n".join(c["text"] for c in chunks)
    system_prompt = (
        "You are a helpful assistant. Answer the user's question based "
        "only on the provided context. If the context doesn't contain "
        "enough information, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return generate(system_prompt, user_prompt)


# ── Full pipeline ────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Run the full contextual retrieval pipeline."""
    query_embedding = embed_query(query)
    semantic_results = semantic_search(query_embedding)
    bm25_results = bm25_search(query)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    # NOTE: With larger datasets, cap fused results before reranking
    # (e.g. fused[:50]) since reranking is the most expensive step.
    reranked = rerank(query, fused, top_n=top_k)
    return reranked


def retrieve_and_answer(query: str, top_k: int = TOP_K) -> str:
    """Retrieve chunks and generate an answer. Returns None if irrelevant."""
    query_embedding = embed_query(query)

    # Start LM Studio headless while retrieval runs
    start_server()

    semantic_results = semantic_search(query_embedding)
    bm25_results = bm25_search(query)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    # NOTE: With larger datasets, cap fused results before reranking
    # (e.g. fused[:50]) since reranking is the most expensive step.
    reranked = rerank(query, fused, top_n=top_k)

    if not reranked or reranked[0]["relevance_score"] < RELEVANCE_THRESHOLD:
        stop_server()
        return None

    try:
        answer = generate_answer(query, reranked)
    finally:
        stop_server()
    return answer


def main() -> None:
    query = input("Enter query: ")
    answer = retrieve_and_answer(query)
    if answer is None:
        print("Your question doesn't seem relevant to the available content.")
    else:
        print(f"\n{answer}")


if __name__ == "__main__":
    main()
