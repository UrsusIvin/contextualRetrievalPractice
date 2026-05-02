"""
embedding.py – Generate vector embeddings for contextualized chunks using Cohere
and store them in a local Qdrant vector database.

Reads contextualized_chunks.txt, sends each chunk (context + text) to
Cohere's Embed API, and upserts the embeddings into a Qdrant collection.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
MODEL = "embed-english-v3.0"
INPUT_TYPE = "search_document"
COLLECTION = "contextualized_chunks"
DIMENSION = 1024


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


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Send chunks to Cohere and return their embeddings."""
    co = cohere.ClientV2(api_key=API_KEY)
    response = co.embed(
        texts=chunks,
        model=MODEL,
        input_type=INPUT_TYPE,
        embedding_types=["float"],
    )
    return response.embeddings.float_


def store_in_qdrant(chunks: list[str], embeddings: list[list[float]]) -> None:
    """Create a Qdrant collection and upsert chunk embeddings."""
    client = QdrantClient(path="./qdrant_db")

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i + 1,
            vector=emb,
            payload={"text": chunk},
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    client.close()


def main() -> None:
    chunks = load_contextualized_chunks()
    print(f"Loaded {len(chunks)} contextualized chunks.")

    embeddings = embed_chunks(chunks)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.")

    store_in_qdrant(chunks, embeddings)
    print(f"Stored {len(chunks)} vectors in Qdrant collection '{COLLECTION}' (./qdrant_db)")


if __name__ == "__main__":
    main()
