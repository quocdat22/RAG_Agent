import hashlib
import logging
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core import Document, SimpleDirectoryReader

# Suppress noisy PDF library warnings
warnings.filterwarnings('ignore', message='.*Cannot set gray non-stroke color.*')
warnings.filterwarnings('ignore', message='.*invalid float value.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pdfplumber')
warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer')

from rag.embeddings import embed_texts
from rag.exceptions import IngestionError, EmbeddingError, StorageError
from rag.vector_store import get_vector_store
from ingestion.chunking import smart_chunk_documents
from ingestion.metadata_schema import normalize_metadata, validate_metadata, MetadataFields

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to calculate hash for {file_path}: {e}")
        # Return empty hash if file cannot be read
        return ""


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get file metadata including hash and modification time.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file_hash and file_mtime
    """
    metadata = {}
    
    try:
        if os.path.exists(file_path):
            # Get modification time
            mtime = os.path.getmtime(file_path)
            metadata["file_mtime"] = mtime
            
            # Calculate hash
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                metadata["file_hash"] = file_hash
    except Exception as e:
        logger.warning(f"Failed to get metadata for {file_path}: {e}")
    
    return metadata


def load_documents(folder_path: str) -> List[Document]:
    """Load documents from a folder using LlamaIndex readers."""
    # Suppress warnings during document loading (PDF processing can be noisy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    
    # Group nodes by document to calculate file metadata once per document
    document_metadata_cache = {}
    
    # Normalize and validate metadata for consistent identification
    for idx, node in enumerate(nodes):
        raw_metadata = node.metadata or {}
        # Normalize metadata to use standard field names
        normalized_metadata = normalize_metadata(raw_metadata)
        
        # Add chunk index if not present
        if "chunk_index" not in normalized_metadata:
            normalized_metadata["chunk_index"] = idx
        
        # Get file path and calculate file metadata if not already cached
        file_path = normalized_metadata.get(MetadataFields.FILE_PATH, "unknown")
        if file_path != "unknown" and file_path not in document_metadata_cache:
            # Try to get actual file path from the document
            # File path might be relative, try to resolve it
            try:
                # Check if file exists at the path
                if os.path.exists(file_path):
                    file_meta = get_file_metadata(file_path)
                    document_metadata_cache[file_path] = file_meta
                else:
                    # Try relative to folder_path
                    full_path = os.path.join(folder_path, file_path)
                    if os.path.exists(full_path):
                        file_meta = get_file_metadata(full_path)
                        document_metadata_cache[file_path] = file_meta
                    else:
                        # Try to find the file in the folder
                        path_obj = Path(folder_path)
                        found_file = None
                        for f in path_obj.rglob(Path(file_path).name):
                            if f.is_file():
                                found_file = str(f)
                                break
                        if found_file:
                            file_meta = get_file_metadata(found_file)
                            document_metadata_cache[file_path] = file_meta
                        else:
                            document_metadata_cache[file_path] = {}
            except Exception as e:
                logger.warning(f"Could not get file metadata for {file_path}: {e}")
                document_metadata_cache[file_path] = {}
        
        # Add file metadata to chunk metadata
        if file_path in document_metadata_cache:
            file_meta = document_metadata_cache[file_path]
            if file_meta.get("file_hash"):
                normalized_metadata[MetadataFields.FILE_HASH] = file_meta["file_hash"]
            if file_meta.get("file_mtime") is not None:
                normalized_metadata[MetadataFields.FILE_MTIME] = file_meta["file_mtime"]
        
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
        stats = vs.upsert_documents(texts=texts, metadatas=metadatas, embeddings=embeddings)
        logger.info(
            f"Ingestion complete: {stats['updated_count']} updated, "
            f"{stats['new_count']} new, {stats['skipped_count']} skipped"
        )
        # Return total chunks processed (updated + new)
        return stats['updated_count'] + stats['new_count']
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


