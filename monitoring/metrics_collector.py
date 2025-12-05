"""
Metrics collector for tracking query performance, retrieval quality, and chunk metrics.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from rag.retriever import CandidateChunk

from monitoring.metrics_store import get_metrics_store


class MetricsCollector:
    """Collects and stores metrics for RAG queries."""

    def __init__(self):
        self.store = get_metrics_store()
        self.query_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.retrieval_start: Optional[float] = None
        self.reranking_start: Optional[float] = None
        self.llm_start: Optional[float] = None

    def start_query(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Start tracking a new query. Returns query_id."""
        self.query_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.query = query
        self.conversation_id = conversation_id
        return self.query_id

    def start_retrieval(self):
        """Mark the start of retrieval phase."""
        self.retrieval_start = time.time()

    def end_retrieval(self, candidates: List[CandidateChunk]):
        """Mark the end of retrieval phase and log retrieval metrics."""
        if not self.retrieval_start:
            return

        retrieval_latency_ms = (time.time() - self.retrieval_start) * 1000

        # Calculate retrieval statistics
        if candidates:
            scores = [c.score for c in candidates]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
        else:
            avg_score = min_score = max_score = 0.0

        # Save retrieval metrics for dense and sparse (if available)
        # For hybrid retrieval, we'll log the combined result
        retrieval_id = str(uuid.uuid4())
        self.store.save_retrieval_metric(
            retrieval_id=retrieval_id,
            query_metric_id=self.query_id,
            retrieval_method="hybrid",
            num_results=len(candidates),
            avg_score=avg_score,
            min_score=min_score,
            max_score=max_score,
        )

        # Log individual chunk quality
        for idx, chunk in enumerate(candidates):
            chunk_id = str(uuid.uuid4())
            self.store.save_chunk_quality(
                chunk_id=chunk_id,
                query_metric_id=self.query_id,
                chunk_index=idx,
                score=chunk.score,
                chunk_text_preview=chunk.text[:500] if chunk.text else None,
                file_path=chunk.metadata.get("file_path") if chunk.metadata else None,
                chunk_index_in_doc=chunk.metadata.get("chunk_index") if chunk.metadata else None,
                retrieval_method="hybrid",
                is_final_chunk=False,
            )

        self.retrieval_latency_ms = retrieval_latency_ms
        self.num_candidates = len(candidates)

    def start_reranking(self):
        """Mark the start of reranking phase."""
        self.reranking_start = time.time()

    def end_reranking(self, top_chunks: List[CandidateChunk]):
        """Mark the end of reranking phase."""
        if not self.reranking_start:
            return

        reranking_latency_ms = (time.time() - self.reranking_start) * 1000
        self.reranking_latency_ms = reranking_latency_ms
        self.num_final_chunks = len(top_chunks)

        # Update chunk quality to mark final chunks
        # Note: In a real implementation, you'd update existing records
        # For simplicity, we'll just log the final chunks separately
        for idx, chunk in enumerate(top_chunks):
            chunk_id = str(uuid.uuid4())
            self.store.save_chunk_quality(
                chunk_id=chunk_id,
                query_metric_id=self.query_id,
                chunk_index=idx,
                score=chunk.score,
                chunk_text_preview=chunk.text[:500] if chunk.text else None,
                file_path=chunk.metadata.get("file_path") if chunk.metadata else None,
                chunk_index_in_doc=chunk.metadata.get("chunk_index") if chunk.metadata else None,
                retrieval_method="reranked",
                is_final_chunk=True,
            )

    def start_llm(self):
        """Mark the start of LLM generation phase."""
        self.llm_start = time.time()

    def end_llm(self):
        """Mark the end of LLM generation phase."""
        if not self.llm_start:
            return

        llm_latency_ms = (time.time() - self.llm_start) * 1000
        self.llm_latency_ms = llm_latency_ms

    def end_query(self):
        """End query tracking and save all metrics."""
        if not self.start_time or not self.query_id:
            return

        try:
            total_latency_ms = (time.time() - self.start_time) * 1000

            self.store.save_query_metric(
                query_id=self.query_id,
                query=self.query,
                conversation_id=self.conversation_id,
                total_latency_ms=total_latency_ms,
                retrieval_latency_ms=getattr(self, "retrieval_latency_ms", None),
                reranking_latency_ms=getattr(self, "reranking_latency_ms", None),
                llm_latency_ms=getattr(self, "llm_latency_ms", None),
                num_candidates=getattr(self, "num_candidates", None),
                num_final_chunks=getattr(self, "num_final_chunks", None),
            )
        except Exception as e:
            # Log error but don't fail the query
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to save query metrics: {e}", exc_info=True)

    def calculate_precision_recall(
        self,
        relevant_chunks: List[str],
        retrieved_chunks: List[str],
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            relevant_chunks: List of chunk IDs/texts that are actually relevant (ground truth)
            retrieved_chunks: List of chunk IDs/texts that were retrieved
        
        Returns:
            Dictionary with precision, recall, and f1_score
        """
        if not retrieved_chunks:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        if not relevant_chunks:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # Convert to sets for easier comparison
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_chunks)

        # Calculate true positives (chunks that are both relevant and retrieved)
        true_positives = len(relevant_set & retrieved_set)

        # Precision = TP / (TP + FP) = TP / retrieved
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0

        # Recall = TP / (TP + FN) = TP / relevant
        recall = true_positives / len(relevant_set) if relevant_set else 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Save precision/recall metrics
        pr_id = str(uuid.uuid4())
        self.store.save_precision_recall(
            pr_id=pr_id,
            query_metric_id=self.query_id,
            relevant_chunks=list(relevant_chunks),
            retrieved_chunks=list(retrieved_chunks),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }


def collect_retrieval_metrics(
    query: str,
    candidates: List[CandidateChunk],
    retrieval_method: str = "hybrid",
) -> Dict[str, Any]:
    """
    Helper function to collect retrieval metrics.
    
    Returns a dictionary with retrieval statistics.
    """
    if not candidates:
        return {
            "num_results": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
        }

    scores = [c.score for c in candidates]
    return {
        "num_results": len(candidates),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "retrieval_method": retrieval_method,
    }

