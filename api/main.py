from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import get_settings
from ingestion.pipeline import run_ingestion
from rag.pipeline import AnswerWithSources, answer_query
from storage.conversation_store import get_conversation_store


class IngestRequest(BaseModel):
    folder_path: str


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    use_history: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: str
    message_id: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: Optional[int] = None


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict]


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class CreateConversationResponse(BaseModel):
    conversation_id: str
    title: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="RAG Agent MVP")

    @app.post("/ingest-folder")
    def ingest_folder(payload: IngestRequest):
        try:
            count = run_ingestion(payload.folder_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {"inserted_chunks": count}

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest):
        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        store = get_conversation_store()
        
        # Create conversation if not provided
        conversation_id = payload.conversation_id
        if not conversation_id:
            conversation_id = store.create_conversation()

        # Save user message
        user_message_id = store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=payload.query,
        )

        # Get answer from RAG
        result: AnswerWithSources = answer_query(
            payload.query,
            conversation_id=conversation_id,
            use_history=payload.use_history,
        )

        # Save assistant message
        assistant_message_id = store.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.answer,
            sources=result.sources,
        )

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
        )

    @app.get("/conversations", response_model=list[ConversationResponse])
    def list_conversations(limit: int = 50, offset: int = 0):
        store = get_conversation_store()
        conversations = store.list_conversations(limit=limit, offset=offset)
        return [
            ConversationResponse(
                id=conv["id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=conv.get("message_count", 0),
            )
            for conv in conversations
        ]

    @app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
    def get_conversation(conversation_id: str):
        store = get_conversation_store()
        conversation = store.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = store.get_messages(conversation_id)
        return ConversationDetailResponse(
            id=conversation["id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            messages=messages,
        )

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        store = get_conversation_store()
        success = store.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}

    @app.post("/conversations", response_model=CreateConversationResponse)
    def create_conversation(payload: CreateConversationRequest):
        store = get_conversation_store()
        conversation_id = store.create_conversation(title=payload.title)
        conversation = store.get_conversation(conversation_id)
        return CreateConversationResponse(
            conversation_id=conversation_id,
            title=conversation["title"],
        )

    return app


app = create_app()


