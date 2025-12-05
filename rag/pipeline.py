import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.exceptions import LLMError, RetrievalError, RAGAgentException
from rag.llm import generate_answer
from rag.prompts import SYSTEM_PROMPT
from rag.reranker import rerank
from rag.retriever import CandidateChunk, hybrid_retrieve
from storage.conversation_store import get_conversation_store
from monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class AnswerWithSources:
    answer: str
    sources: List[Dict[str, Any]]


def answer_query(
    query: str,
    conversation_id: Optional[str] = None,
    use_history: bool = False,
) -> AnswerWithSources:
    """
    Answer a query using RAG pipeline with improved error handling and metrics collection.
    
    Raises:
        RetrievalError: If retrieval fails
        LLMError: If LLM generation fails
        RAGAgentException: For other RAG-related errors
    """
    if not query or not query.strip():
        raise RetrievalError(
            "Empty query provided",
            user_message="Câu hỏi không được để trống. Vui lòng nhập câu hỏi của bạn."
        )
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    metrics.start_query(query, conversation_id)
    
    store = None
    allowed_docs: Optional[List[str]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

    try:
        if conversation_id:
            store = get_conversation_store()
            selected_docs = store.get_selected_documents(conversation_id)
            if selected_docs:
                allowed_docs = selected_docs
            if use_history:
                conversation_history = store.get_conversation_history(
                    conversation_id, max_messages=10
                )
        elif use_history:
            # Can't fetch history without a conversation, so skip.
            use_history = False
            logger.warning("use_history=True but no conversation_id provided, ignoring history")
    except Exception as e:
        logger.error(f"Error fetching conversation data: {str(e)}", exc_info=True)
        # Continue without history/selected docs if there's an error

    # Track retrieval phase
    metrics.start_retrieval()
    try:
        candidates: List[CandidateChunk] = hybrid_retrieve(
            query, allowed_file_paths=allowed_docs
        )
    except Exception as e:
        metrics.end_query()  # Save metrics even on error
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
        raise RetrievalError(
            f"Failed to retrieve documents: {str(e)}",
            user_message=(
                "Không thể tìm kiếm thông tin trong tài liệu. "
                "Có thể do lỗi kết nối cơ sở dữ liệu hoặc không có tài liệu phù hợp. "
                "Vui lòng thử lại sau."
            ),
            details={"error_type": type(e).__name__}
        ) from e
    finally:
        metrics.end_retrieval(candidates)

    if not candidates:
        metrics.end_query()
        logger.warning(f"No candidates found for query: {query[:50]}...")
        return AnswerWithSources(
            answer="Không tìm thấy thông tin liên quan trong tài liệu nội bộ.",
            sources=[]
        )

    # Track reranking phase
    metrics.start_reranking()
    try:
        top_chunks = rerank(query, candidates, top_n=5)
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}", exc_info=True)
        # Fallback: use top candidates without reranking
        logger.warning("Falling back to top candidates without reranking")
        top_chunks = candidates[:5]
    finally:
        metrics.end_reranking(top_chunks)

    contexts = [c.text for c in top_chunks]
    
    # Track LLM generation phase
    metrics.start_llm()
    try:
        answer = generate_answer(query, contexts, conversation_history=conversation_history)
    except LLMError:
        metrics.end_query()  # Save metrics even on error
        # Re-raise LLMError as-is (already has user-friendly message)
        raise
    except Exception as e:
        metrics.end_query()  # Save metrics even on error
        logger.error(f"LLM generation failed: {str(e)}", exc_info=True)
        # Wrap unexpected errors
        raise LLMError(
            f"Unexpected error during LLM generation: {str(e)}",
            user_message=(
                "Xảy ra lỗi không mong đợi khi tạo câu trả lời. "
                "Vui lòng thử lại hoặc liên hệ quản trị viên."
            ),
            details={"error_type": type(e).__name__}
        ) from e
    finally:
        metrics.end_llm()

    # Save all metrics
    metrics.end_query()

    sources = [c.metadata for c in top_chunks]
    return AnswerWithSources(answer=answer, sources=sources)


