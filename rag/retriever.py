from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.embeddings import embed_texts
from rag.sparse_index import SparseIndex
from rag.vector_store import get_collection


@dataclass
class CandidateChunk:
    text: str
    metadata: Dict[str, Any]
    score: float


def dense_retrieve(
    query: str,
    k: int = 20,
    allowed_file_paths: Optional[List[str]] = None,
) -> List[CandidateChunk]:
    collection = get_collection()
    [query_emb] = embed_texts([query])
    where = None
    if allowed_file_paths:
        # Normalize where clause based on number of documents
        if len(allowed_file_paths) == 1:
            where = {"file_path": allowed_file_paths[0]}
        else:
            where = {"file_path": {"$in": allowed_file_paths}}

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where=where,
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or [0.0] * len(docs)

    return [
        CandidateChunk(text=doc, metadata=meta or {}, score=float(-dist))
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def sparse_retrieve(query: str, all_texts: List[str], all_metadatas: List[Dict[str, Any]], k: int = 20) -> List[CandidateChunk]:
    index = SparseIndex(all_texts)
    top_idx = index.search(query, k=k)
    return [
        CandidateChunk(
            text=all_texts[i],
            metadata=all_metadatas[i] if i < len(all_metadatas) else {},
            score=float(k - rank),
        )
        for rank, i in enumerate(top_idx)
    ]


def hybrid_retrieve(
    query: str,
    k_dense: int = 20,
    k_sparse: int = 20,
    allowed_file_paths: Optional[List[str]] = None,
) -> List[CandidateChunk]:
    dense = dense_retrieve(query, k=k_dense, allowed_file_paths=allowed_file_paths)

    # For MVP, sparse over dense results' texts
    texts = [c.text for c in dense]
    metas = [c.metadata for c in dense]
    sparse = sparse_retrieve(query, texts, metas, k=k_sparse)

    combined: Dict[str, CandidateChunk] = {}
    for c in dense + sparse:
        key = c.text
        if key not in combined or c.score > combined[key].score:
            combined[key] = c

    return sorted(combined.values(), key=lambda c: c.score, reverse=True)


