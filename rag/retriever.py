from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple

from rag.embeddings import embed_texts
from rag.sparse_index import SparseIndex
from rag.vector_store import get_collection
from ingestion.metadata_schema import normalize_metadata, MetadataFields


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
        # Use normalized file_path field for consistent querying
        if len(allowed_file_paths) == 1:
            where = {MetadataFields.FILE_PATH: allowed_file_paths[0]}
        else:
            where = {MetadataFields.FILE_PATH: {"$in": allowed_file_paths}}

    res = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where=where,
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or [0.0] * len(docs)

    # Normalize metadata in results for consistency
    return [
        CandidateChunk(
            text=doc, 
            metadata=normalize_metadata(meta or {}), 
            score=float(-dist)
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def _get_all_corpus_texts(
    allowed_file_paths: Optional[List[str]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Get all texts and metadatas from the collection, optionally filtered by file paths.
    
    Returns:
        Tuple of (texts, metadatas) lists
    """
    collection = get_collection()
    where = None
    if allowed_file_paths:
        if len(allowed_file_paths) == 1:
            where = {MetadataFields.FILE_PATH: allowed_file_paths[0]}
        else:
            where = {MetadataFields.FILE_PATH: {"$in": allowed_file_paths}}
    
    results = collection.get(where=where)
    texts = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    
    # Normalize all metadatas
    normalized_metadatas = [
        normalize_metadata(meta or {}) for meta in metadatas
    ]
    
    return texts, normalized_metadatas


def sparse_retrieve(
    query: str, 
    all_texts: List[str], 
    all_metadatas: List[Dict[str, Any]], 
    k: int = 20
) -> List[CandidateChunk]:
    """
    Perform sparse retrieval using BM25 on the given corpus.
    
    Returns candidates with BM25 scores (higher is better).
    """
    if not all_texts:
        return []
    
    index = SparseIndex(all_texts)
    tokenized_query = query.split()
    scores = index.bm25.get_scores(tokenized_query)
    
    # Get top k indices sorted by score
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]
    
    # Return candidates with actual BM25 scores
    return [
        CandidateChunk(
            text=all_texts[i],
            metadata=normalize_metadata(all_metadatas[i] if i < len(all_metadatas) else {}),
            score=float(scores[i]),  # Use actual BM25 score
        )
        for i in ranked_indices
    ]


def _reciprocal_rank_fusion(
    dense_results: List[CandidateChunk],
    sparse_results: List[CandidateChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[CandidateChunk]:
    """
    Combine results using Reciprocal Rank Fusion (RRF).
    
    RRF score = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)
    
    Args:
        dense_results: Dense retrieval results
        sparse_results: Sparse retrieval results
        k: RRF constant (typically 60)
        dense_weight: Weight for dense scores
        sparse_weight: Weight for sparse scores
        
    Returns:
        Combined and ranked candidates
    """
    # Create text -> rank mappings
    dense_ranks: Dict[str, int] = {
        c.text: rank + 1 for rank, c in enumerate(dense_results)
    }
    sparse_ranks: Dict[str, int] = {
        c.text: rank + 1 for rank, c in enumerate(sparse_results)
    }
    
    # Get all unique texts
    all_texts = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    
    # Calculate RRF scores
    fused_candidates: Dict[str, CandidateChunk] = {}
    for text in all_texts:
        dense_rank = dense_ranks.get(text, k + 1)
        sparse_rank = sparse_ranks.get(text, k + 1)
        
        rrf_score = (
            dense_weight / (k + dense_rank) + 
            sparse_weight / (k + sparse_rank)
        )
        
        # Get metadata from whichever result has it (prefer dense)
        if text in dense_ranks:
            metadata = dense_results[dense_ranks[text] - 1].metadata
        elif text in sparse_ranks:
            metadata = sparse_results[sparse_ranks[text] - 1].metadata
        else:
            metadata = {}
        
        fused_candidates[text] = CandidateChunk(
            text=text,
            metadata=metadata,
            score=rrf_score,
        )
    
    return sorted(fused_candidates.values(), key=lambda c: c.score, reverse=True)


def _weighted_sum_fusion(
    dense_results: List[CandidateChunk],
    sparse_results: List[CandidateChunk],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> List[CandidateChunk]:
    """
    Combine results using weighted sum of normalized scores.
    
    Final score = dense_weight * normalized_dense_score + sparse_weight * normalized_sparse_score
    
    Args:
        dense_results: Dense retrieval results
        sparse_results: Sparse retrieval results
        dense_weight: Weight for dense scores (should sum to 1.0 with sparse_weight)
        sparse_weight: Weight for sparse scores
        
    Returns:
        Combined and ranked candidates
    """
    # Normalize scores to [0, 1] range
    def normalize_scores(results: List[CandidateChunk]) -> Dict[str, float]:
        if not results:
            return {}
        min_score = min(c.score for c in results)
        max_score = max(c.score for c in results)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        return {
            c.text: (c.score - min_score) / score_range
            for c in results
        }
    
    dense_norm = normalize_scores(dense_results)
    sparse_norm = normalize_scores(sparse_results)
    
    # Get all unique texts
    all_texts = set(dense_norm.keys()) | set(sparse_norm.keys())
    
    # Calculate weighted sum scores
    fused_candidates: Dict[str, CandidateChunk] = {}
    for text in all_texts:
        dense_score = dense_norm.get(text, 0.0)
        sparse_score = sparse_norm.get(text, 0.0)
        
        fused_score = dense_weight * dense_score + sparse_weight * sparse_score
        
        # Get metadata from whichever result has it (prefer dense)
        dense_candidate = next((c for c in dense_results if c.text == text), None)
        sparse_candidate = next((c for c in sparse_results if c.text == text), None)
        
        metadata = (
            dense_candidate.metadata if dense_candidate
            else (sparse_candidate.metadata if sparse_candidate else {})
        )
        
        fused_candidates[text] = CandidateChunk(
            text=text,
            metadata=metadata,
            score=fused_score,
        )
    
    return sorted(fused_candidates.values(), key=lambda c: c.score, reverse=True)


def hybrid_retrieve(
    query: str,
    k_dense: int = 20,
    k_sparse: int = 20,
    k_final: Optional[int] = None,
    allowed_file_paths: Optional[List[str]] = None,
    fusion_method: Literal["rrf", "weighted_sum"] = "rrf",
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_k: int = 60,
) -> List[CandidateChunk]:
    """
    Perform hybrid retrieval combining dense and sparse search.
    
    Args:
        query: Search query
        k_dense: Number of results from dense retrieval
        k_sparse: Number of results from sparse retrieval
        k_final: Final number of results to return (None = return all fused results)
        allowed_file_paths: Optional list of file paths to filter by
        fusion_method: "rrf" for Reciprocal Rank Fusion or "weighted_sum" for weighted sum
        dense_weight: Weight for dense retrieval (used in both fusion methods)
        sparse_weight: Weight for sparse retrieval (used in both fusion methods)
        rrf_k: RRF constant (only used when fusion_method="rrf")
        
    Returns:
        Combined and ranked candidates
    """
    # Get full corpus for sparse search
    all_texts, all_metadatas = _get_all_corpus_texts(allowed_file_paths=allowed_file_paths)
    
    # Perform both retrievals independently on full corpus
    dense = dense_retrieve(query, k=k_dense, allowed_file_paths=allowed_file_paths)
    sparse = sparse_retrieve(query, all_texts, all_metadatas, k=k_sparse)
    
    # Fuse results
    if fusion_method == "rrf":
        fused = _reciprocal_rank_fusion(
            dense, sparse, k=rrf_k, 
            dense_weight=dense_weight, 
            sparse_weight=sparse_weight
        )
    else:  # weighted_sum
        fused = _weighted_sum_fusion(
            dense, sparse,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
    
    # Return top k_final results if specified
    if k_final is not None:
        return fused[:k_final]
    return fused


