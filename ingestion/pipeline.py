import logging
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

from rag.embeddings import embed_texts
from rag.exceptions import IngestionError, EmbeddingError, StorageError
from rag.vector_store import get_vector_store
from ingestion.chunking import smart_chunk_documents
from ingestion.metadata_schema import normalize_metadata, validate_metadata

logger = logging.getLogger(__name__)


def load_documents(folder_path: str) -> List[Document]:
    """Load documents from a folder using LlamaIndex readers."""
    reader = SimpleDirectoryReader(input_dir=folder_path, recursive=True)
    docs = reader.load_data()
    return docs


def preprocess_and_chunk(documents: List[Document]):
    """
    Clean and chunk documents into nodes using smart chunking strategy.
    Preserves document structure (headers, lists, code blocks, tables).
    """
    return smart_chunk_documents(documents)


def run_ingestion(folder_path: str) -> int:
    """
    Full ingestion pipeline:
    - load documents
    - preprocess + chunk
    - embed
    - store in Chroma

    Returns number of chunks inserted.
    
    Raises:
        IngestionError: If ingestion fails at any step
    """
    path = Path(folder_path)
    if not path.exists() or not path.is_dir():
        raise IngestionError(
            f"Folder not found: {folder_path}",
            user_message=f"Không tìm thấy thư mục: {folder_path}. Vui lòng kiểm tra đường dẫn."
        )

    try:
        documents = load_documents(folder_path)
    except Exception as e:
        logger.error(f"Failed to load documents: {str(e)}", exc_info=True)
        raise IngestionError(
            f"Failed to load documents: {str(e)}",
            user_message=(
                "Không thể đọc tài liệu từ thư mục. "
                "Vui lòng kiểm tra định dạng file được hỗ trợ (PDF, DOCX, XLSX, HTML, MD, TXT)."
            ),
            details={"error_type": type(e).__name__}
        ) from e

    if not documents:
        logger.warning(f"No documents found in folder: {folder_path}")
        return 0

    try:
        nodes = preprocess_and_chunk(documents)
    except Exception as e:
        logger.error(f"Failed to preprocess documents: {str(e)}", exc_info=True)
        raise IngestionError(
            f"Failed to preprocess documents: {str(e)}",
            user_message="Không thể xử lý tài liệu. Vui lòng kiểm tra định dạng file và thử lại."
        ) from e

    texts = [n.get_content() for n in nodes]
    metadatas = []
    
    # Normalize and validate metadata for consistent identification
    for idx, node in enumerate(nodes):
        raw_metadata = node.metadata or {}
        # Normalize metadata to use standard field names
        normalized_metadata = normalize_metadata(raw_metadata)
        
        # Add chunk index if not present
        if "chunk_index" not in normalized_metadata:
            normalized_metadata["chunk_index"] = idx
        
        # Validate metadata
        is_valid, error_msg = validate_metadata(normalized_metadata, strict=False)
        if not is_valid:
            logger.warning(f"Metadata validation warning for chunk {idx}: {error_msg}")
        
        metadatas.append(normalized_metadata)

    try:
        embeddings = embed_texts(texts)
    except EmbeddingError:
        # Re-raise EmbeddingError as-is
        raise
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
        raise IngestionError(
            f"Failed to generate embeddings: {str(e)}",
            user_message=(
                "Không thể tạo embedding cho tài liệu. "
                "Có thể do lỗi kết nối dịch vụ embedding. Vui lòng thử lại sau."
            ),
            details={"error_type": type(e).__name__}
        ) from e

    if len(embeddings) != len(texts):
        logger.error(
            f"Embedding count mismatch: {len(texts)} texts but {len(embeddings)} embeddings"
        )
        raise IngestionError(
            f"Embedding count mismatch: {len(texts)} texts but {len(embeddings)} embeddings",
            user_message="Lỗi khi tạo embedding. Vui lòng thử lại."
        )

    try:
        vs = get_vector_store()
        vs.add_documents(texts=texts, metadatas=metadatas, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Failed to store documents: {str(e)}", exc_info=True)
        raise IngestionError(
            f"Failed to store documents: {str(e)}",
            user_message=(
                "Không thể lưu tài liệu vào cơ sở dữ liệu. "
                "Vui lòng kiểm tra kết nối và thử lại."
            ),
            details={"error_type": type(e).__name__}
        ) from e

    logger.info(f"Successfully ingested {len(nodes)} chunks from {folder_path}")
    return len(nodes)


