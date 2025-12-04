import logging
from pathlib import Path
from typing import List
import re

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from rag.embeddings import embed_texts
from rag.exceptions import IngestionError, EmbeddingError, StorageError
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def detect_and_format_tables(text: str) -> str:
    """
    Phát hiện và format lại dữ liệu dạng bảng trong text.
    Tìm các pattern như: nhiều dòng có cùng số cột, dữ liệu có tab/space phân cách.
    """
    lines = text.split('\n')
    table_candidates = []
    current_table = []
    
    for line in lines:
        # Phát hiện dòng có nhiều cột (tab, pipe, hoặc nhiều space)
        parts = re.split(r'\t|\|', line.strip())
        if len(parts) >= 2 and any(part.strip() for part in parts):
            # Có thể là hàng của bảng
            if not current_table:
                current_table = [line]
            else:
                # Kiểm tra số cột có khớp không
                prev_parts = re.split(r'\t|\|', current_table[-1].strip())
                if len(parts) == len(prev_parts):
                    current_table.append(line)
                else:
                    # Kết thúc bảng hiện tại
                    if len(current_table) >= 2:
                        table_candidates.append(current_table)
                    current_table = [line] if len(parts) >= 2 else []
        else:
            if current_table and len(current_table) >= 2:
                table_candidates.append(current_table)
            current_table = []
    
    if current_table and len(current_table) >= 2:
        table_candidates.append(current_table)
    
    # Format lại thành markdown table
    result = text
    for table in table_candidates:
        original = '\n'.join(table)
        # Chuyển thành markdown table
        formatted_rows = []
        for i, row in enumerate(table):
            parts = re.split(r'\t|\|', row.strip())
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                formatted_row = '| ' + ' | '.join(parts) + ' |'
                formatted_rows.append(formatted_row)
                if i == 0:
                    # Thêm separator
                    separator = '| ' + ' | '.join(['---'] * len(parts)) + ' |'
                    formatted_rows.append(separator)
        
        if formatted_rows:
            formatted_table = '\n'.join(formatted_rows)
            result = result.replace(original, f"\n\n{formatted_table}\n\n")
    
    return result


def load_documents(folder_path: str) -> List[Document]:
    """Load documents from a folder using LlamaIndex readers."""
    reader = SimpleDirectoryReader(input_dir=folder_path, recursive=True)
    docs = reader.load_data()
    return docs


def preprocess_and_chunk(documents: List[Document]):
    """Clean and chunk documents into nodes."""
    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Xử lý và format tables trong mỗi node
    for node in nodes:
        content = node.get_content()
        formatted_content = detect_and_format_tables(content)
        node.set_content(formatted_content)
    
    return nodes


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
    
    # Ensure metadata has file_path for consistent identification
    for node in nodes:
        metadata = node.metadata or {}
        # Normalize file_path - use file_path, file_name, filename, or source
        if not metadata.get("file_path"):
            metadata["file_path"] = (
                metadata.get("file_name") or 
                metadata.get("filename") or 
                metadata.get("source") or
                "unknown"
            )
        metadatas.append(metadata)

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


