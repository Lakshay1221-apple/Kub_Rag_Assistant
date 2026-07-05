"""Qdrant-backed retrieval service."""

import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embeddings import embed_query


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client on demand."""

    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )


def search_enterprise_knowledge(query: str, limit: int = 15) -> list[dict]:
    """
    Search the configured Qdrant collection for chunks relevant to the query.

    Returns dictionaries with a stable ``content`` key for the retriever node,
    plus source metadata when available.
    """

    if not query.strip():
        return []

    query_vector = embed_query(query)
    client = get_qdrant_client()

    try:
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logfire.error(f"Qdrant search failed: {exc}")
        raise

    results: list[dict] = []
    for point in response.points:
        payload = point.payload or {}
        content = payload.get("text") or payload.get("content") or ""
        if not content:
            continue

        results.append(
            {
                "content": content,
                "score": point.score,
                "filename": payload.get("filename"),
                "source_type": payload.get("source_type"),
                "chunk_index": payload.get("chunk_index"),
            }
        )

    return results
