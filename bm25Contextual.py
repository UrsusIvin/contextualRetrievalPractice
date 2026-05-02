"""
bm25Contextual.py – Build a BM25 index over contextualized chunks
and store the data in a local Qdrant collection.

Reads contextualized_chunks.txt, tokenizes each chunk, computes
IDF values, and stores tokenized docs + BM25 metadata in Qdrant.
"""

import math
import re
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "bm25_chunks"
# BM25 doesn't use real vector search, but Qdrant needs a vector config.
# We store a dummy 1-D vector and use the payload for BM25 scoring.
DUMMY_DIM = 1


def load_contextualized_chunks(path: str = "contextualized_chunks.txt") -> list[str]:
    """Parse contextualized_chunks.txt into a list of full chunk strings."""
    text = Path(path).read_text(encoding="utf-8")
    raw_chunks = re.split(r"--- Chunk \d+ ---\n", text)
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_data(chunks: list[str], k1: float = 1.5, b: float = 0.75) -> dict:
    """Build BM25 metadata from tokenized chunks.

    Returns a dict with:
    - tokenized docs, doc lengths, average doc length
    - IDF values for all terms
    - term frequency maps per document
    - BM25 parameters k1 and b
    """
    tokenized = [tokenize(chunk) for chunk in chunks]
    doc_count = len(tokenized)
    doc_lengths = [len(doc) for doc in tokenized]
    avgdl = sum(doc_lengths) / doc_count

    df = {}
    for doc in tokenized:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)

    tf_maps = []
    for doc in tokenized:
        tf = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        tf_maps.append(tf)

    return {
        "k1": k1,
        "b": b,
        "avgdl": avgdl,
        "doc_count": doc_count,
        "doc_lengths": doc_lengths,
        "tf_maps": tf_maps,
        "idf": idf,
    }


def store_in_qdrant(chunks: list[str], bm25: dict) -> None:
    """Store chunks and their BM25 metadata in a Qdrant collection."""
    client = QdrantClient(path="./qdrant_db")

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DUMMY_DIM, distance=Distance.COSINE),
    )

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=i + 1,
            vector=[0.0],
            payload={
                "text": chunk,
                "tf_map": bm25["tf_maps"][i],
                "doc_length": bm25["doc_lengths"][i],
            },
        ))

    client.upsert(collection_name=COLLECTION, points=points)

    # Store global BM25 params as a special metadata point (id=0)
    client.upsert(collection_name=COLLECTION, points=[
        PointStruct(
            id=0,
            vector=[0.0],
            payload={
                "is_metadata": True,
                "k1": bm25["k1"],
                "b": bm25["b"],
                "avgdl": bm25["avgdl"],
                "doc_count": bm25["doc_count"],
                "idf": bm25["idf"],
            },
        )
    ])

    client.close()


def main() -> None:
    chunks = load_contextualized_chunks()
    print(f"Loaded {len(chunks)} contextualized chunks.")

    bm25 = build_bm25_data(chunks)
    print(f"Built BM25 index: {len(bm25['idf'])} unique terms, avgdl={bm25['avgdl']:.1f}")

    store_in_qdrant(chunks, bm25)
    print(f"Stored BM25 data in Qdrant collection '{COLLECTION}' (./qdrant_db)")


if __name__ == "__main__":
    main()
