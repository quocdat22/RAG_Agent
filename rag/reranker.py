from typing import List

import cohere

from config.settings import get_settings
from rag.retriever import CandidateChunk


def get_client() -> cohere.Client:
    settings = get_settings()
    if not settings.cohere_api_key:
        raise RuntimeError("Cohere API key is not configured")
    return cohere.Client(api_key=settings.cohere_api_key)


def rerank(query: str, candidates: List[CandidateChunk], top_n: int = 5) -> List[CandidateChunk]:
    if not candidates:
        return []

    client = get_client()
    inputs = [c.text for c in candidates]

    resp = client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=inputs,
        top_n=min(top_n, len(inputs)),
    )

    ranked = []
    for r in resp.results:
        c = candidates[r.index]
        ranked.append(
            CandidateChunk(text=c.text, metadata=c.metadata, score=float(r.relevance_score))
        )

    return ranked


