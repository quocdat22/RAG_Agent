import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from config.settings import get_settings
from ingestion.pipeline import run_ingestion
from rag.exceptions import (
    RAGAgentException,
    LLMError,
    RetrievalError,
    ConfigurationError,
    IngestionError,
    StorageError,
)
from rag.pipeline import AnswerWithSources, answer_query
from storage.conversation_store import get_conversation_store

logger = logging.getLogger(__name__)

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)


def verify_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> bool:
    """
    Verify API key from either Authorization header (Bearer token) or X-API-Key header.
    
    Args:
        authorization: Bearer token from Authorization header
        x_api_key: API key from X-API-Key header
    
    Returns:
        True if API key is valid or not required, False otherwise
    
    Raises:
        HTTPException: If API key is required but invalid or missing
    """
    settings = get_settings()
    
    # If no API key is configured, allow all requests (backward compatibility)
    if not settings.api_key:
        return True
    
    # Check X-API-Key header first (more common for API keys)
    if x_api_key:
        if x_api_key == settings.api_key:
            return True
        else:
            logger.warning(f"Invalid API key provided in X-API-Key header")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Check Authorization header (Bearer token)
    if authorization:
        if authorization.credentials == settings.api_key:
            return True
        else:
            logger.warning(f"Invalid API key provided in Authorization header")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # No API key provided but one is required
    logger.warning("API key required but not provided")
    raise HTTPException(
        status_code=401,
        detail="API key required. Provide it in X-API-Key header or Authorization header (Bearer token)",
        headers={"WWW-Authenticate": "Bearer"},
    )


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

    @app.exception_handler(RAGAgentException)
    async def rag_exception_handler(request: Request, exc: RAGAgentException):
        """Handle RAG Agent custom exceptions."""
        logger.error(
            f"RAGAgentException: {exc.user_message}",
            exc_info=True,
            extra={"error_details": exc.details, "path": request.url.path}
        )
        
        status_code = 500
        if isinstance(exc, ConfigurationError):
            status_code = 503  # Service Unavailable
        elif isinstance(exc, (LLMError, RetrievalError)):
            status_code = 502  # Bad Gateway
        elif isinstance(exc, IngestionError):
            status_code = 400  # Bad Request
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.user_message,
                "error_type": type(exc).__name__,
                "details": exc.details,
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            f"Unexpected error: {str(exc)}",
            exc_info=True,
            extra={"path": request.url.path}
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Đã xảy ra lỗi không mong đợi. Vui lòng thử lại sau.",
                "error_type": type(exc).__name__,
            }
        )

    @app.post("/ingest-folder")
    def ingest_folder(payload: IngestRequest, _: bool = Depends(verify_api_key)):
        try:
            count = run_ingestion(payload.folder_path)
        except ValueError as e:
            raise IngestionError(
                f"Invalid folder path: {str(e)}",
                user_message=f"Đường dẫn thư mục không hợp lệ: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise IngestionError(
                f"Ingestion failed: {str(e)}",
                user_message="Không thể xử lý tài liệu. Vui lòng kiểm tra định dạng file và thử lại."
            ) from e
        return {"inserted_chunks": count}

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest, _: bool = Depends(verify_api_key)):
        if not payload.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Câu hỏi không được để trống"
            )

        try:
            store = get_conversation_store()
        except Exception as e:
            logger.error(f"Failed to get conversation store: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to initialize storage: {str(e)}",
                user_message="Không thể kết nối với cơ sở dữ liệu. Vui lòng thử lại sau."
            ) from e
        
        # Create conversation if not provided
        conversation_id = payload.conversation_id
        if not conversation_id:
            try:
                conversation_id = store.create_conversation()
            except Exception as e:
                logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
                raise StorageError(
                    f"Failed to create conversation: {str(e)}",
                    user_message="Không thể tạo cuộc trò chuyện mới. Vui lòng thử lại sau."
                ) from e

        # Save user message
        try:
            user_message_id = store.add_message(
                conversation_id=conversation_id,
                role="user",
                content=payload.query,
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to save message: {str(e)}",
                user_message="Không thể lưu tin nhắn. Vui lòng thử lại sau."
            ) from e

        # Get answer from RAG (exceptions will be handled by exception handler)
        result: AnswerWithSources = answer_query(
            payload.query,
            conversation_id=conversation_id,
            use_history=payload.use_history,
        )

        # Save assistant message
        try:
            assistant_message_id = store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.answer,
                sources=result.sources,
            )
        except Exception as e:
            logger.error(f"Failed to save assistant message: {str(e)}", exc_info=True)
            # Don't fail the request if we can't save the message, just log it
            assistant_message_id = "unknown"

        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
        )

    @app.get("/conversations", response_model=list[ConversationResponse])
    def list_conversations(limit: int = 50, offset: int = 0, _: bool = Depends(verify_api_key)):
        try:
            store = get_conversation_store()
            conversations = store.list_conversations(limit=limit, offset=offset)
        except Exception as e:
            logger.error(f"Failed to list conversations: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to list conversations: {str(e)}",
                user_message="Không thể tải danh sách cuộc trò chuyện. Vui lòng thử lại sau."
            ) from e
        
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
    def get_conversation(conversation_id: str, _: bool = Depends(verify_api_key)):
        try:
            store = get_conversation_store()
            conversation = store.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail="Không tìm thấy cuộc trò chuyện"
                )
            
            messages = store.get_messages(conversation_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get conversation: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to get conversation: {str(e)}",
                user_message="Không thể tải cuộc trò chuyện. Vui lòng thử lại sau."
            ) from e
        
        return ConversationDetailResponse(
            id=conversation["id"],
            title=conversation["title"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            messages=messages,
        )

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, _: bool = Depends(verify_api_key)):
        try:
            store = get_conversation_store()
            success = store.delete_conversation(conversation_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail="Không tìm thấy cuộc trò chuyện"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to delete conversation: {str(e)}",
                user_message="Không thể xóa cuộc trò chuyện. Vui lòng thử lại sau."
            ) from e
        
        return {"message": "Đã xóa cuộc trò chuyện thành công"}

    @app.post("/conversations", response_model=CreateConversationResponse)
    def create_conversation(payload: CreateConversationRequest, _: bool = Depends(verify_api_key)):
        try:
            store = get_conversation_store()
            conversation_id = store.create_conversation(title=payload.title)
            conversation = store.get_conversation(conversation_id)
        except Exception as e:
            logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
            raise StorageError(
                f"Failed to create conversation: {str(e)}",
                user_message="Không thể tạo cuộc trò chuyện mới. Vui lòng thử lại sau."
            ) from e
        
        return CreateConversationResponse(
            conversation_id=conversation_id,
            title=conversation["title"],
        )

    return app


app = create_app()


