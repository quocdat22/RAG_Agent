import uuid
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb import PersistentClient

from config.settings import get_settings
from ingestion.metadata_schema import normalize_metadata, get_file_path, MetadataFields


_client: Optional[PersistentClient] = None
_collection = None


def _get_client() -> PersistentClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection("documents")
    return _collection


def get_vector_store():
    """Tiny wrapper compatible with our ingestion pipeline."""
    return VectorStore()


class VectorStore:
    def __init__(self) -> None:
        self._collection = get_collection()

    def add_documents(
        self,
        texts: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
        embeddings: Iterable[List[float]],
    ) -> None:
        # Normalize all metadata to ensure consistency
        texts_list = list(texts)
        metadatas_list = list(metadatas)
        normalized_metadatas = []
        ids = []
        
        for idx, metadata in enumerate(metadatas_list):
            # Normalize metadata to use standard field names
            normalized_metadata = normalize_metadata(metadata)
            normalized_metadatas.append(normalized_metadata)
            
            # Create ID using normalized file_path
            file_path = normalized_metadata.get(MetadataFields.FILE_PATH, "unknown")
            chunk_index = normalized_metadata.get(MetadataFields.CHUNK_INDEX, idx)
            ids.append(f"{file_path}_{chunk_index}_{uuid.uuid4().hex[:8]}")
        
        self._collection.add(
            ids=ids,
            documents=texts_list,
            metadatas=normalized_metadatas,
            embeddings=list(embeddings),
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 20,
    ):
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all unique documents from the collection.
        Returns a list of unique documents based on file_path or file_name metadata.
        """
        try:
            # Get all documents from the collection
            results = self._collection.get()
            
            if not results or not results.get("metadatas"):
                return []
            
            # Extract unique documents based on normalized file_path
            seen_docs = {}
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            for idx, metadata in enumerate(metadatas):
                if not metadata:
                    continue
                
                # Normalize metadata to get consistent file_path
                normalized_metadata = normalize_metadata(metadata)
                doc_key = normalized_metadata.get(MetadataFields.FILE_PATH, f"Document_{ids[idx]}")
                
                # If we haven't seen this document, add it
                if doc_key not in seen_docs:
                    seen_docs[doc_key] = {
                        "name": doc_key,
                        "file_path": doc_key,
                        "page": normalized_metadata.get(MetadataFields.PAGE),
                        "chunk_count": 1,
                    }
                else:
                    # Increment chunk count for this document
                    seen_docs[doc_key]["chunk_count"] += 1
            
            return list(seen_docs.values())
        except Exception as e:
            # Return empty list if there's an error (e.g., collection is empty)
            return []

    def get_document_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document identified by file_path.
        Returns a list of chunks with their content, metadata, and IDs.
        
        Uses normalized file_path field for consistent querying.
        """
        try:
            # Query using normalized file_path field
            results = self._collection.get(where={MetadataFields.FILE_PATH: file_path})
            
            if not results or not results.get("ids"):
                return []
            
            chunks = []
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            
            for idx, chunk_id in enumerate(ids):
                # Normalize metadata to ensure consistency
                metadata = normalize_metadata(metadatas[idx] if idx < len(metadatas) else {})
                chunks.append({
                    "id": chunk_id,
                    "content": documents[idx] if idx < len(documents) else "",
                    "metadata": metadata,
                })
            
            return chunks
        except Exception as e:
            # Return empty list if there's an error
            return []

    def delete_document(self, file_path: str) -> int:
        """
        Delete all chunks for a specific document identified by file_path.
        Returns the number of chunks deleted.
        
        Uses normalized file_path field for consistent deletion.
        """
        try:
            # First, get all chunks for this document to count them
            chunks = self.get_document_chunks(file_path)
            if not chunks:
                return 0
            
            chunk_count = len(chunks)
            
            # Delete using normalized file_path field
            try:
                self._collection.delete(where={MetadataFields.FILE_PATH: file_path})
            except Exception as e:
                # Log error but return count we found
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error deleting document chunks: {e}")
            
            return chunk_count
        except Exception as e:
            # Return 0 if deletion fails
            return 0


