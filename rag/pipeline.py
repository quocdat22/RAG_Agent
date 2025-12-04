from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.llm import generate_answer
from rag.prompts import SYSTEM_PROMPT
from rag.reranker import rerank
from rag.retriever import CandidateChunk, hybrid_retrieve
from storage.conversation_store import get_conversation_store


@dataclass
class AnswerWithSources:
    answer: str
    sources: List[Dict[str, Any]]


def answer_query(
    query: str,
    conversation_id: Optional[str] = None,
    use_history: bool = False,
) -> AnswerWithSources:
    store = None
    allowed_docs: Optional[List[str]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

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

    candidates: List[CandidateChunk] = hybrid_retrieve(
        query, allowed_file_paths=allowed_docs
    )
    top_chunks = rerank(query, candidates, top_n=5)

    contexts = [c.text for c in top_chunks]
    
    answer = generate_answer(query, contexts, conversation_history=conversation_history)

    sources = [c.metadata for c in top_chunks]
    return AnswerWithSources(answer=answer, sources=sources)


