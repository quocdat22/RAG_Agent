import uuid
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb import PersistentClient

from config.settings import get_settings


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
        # Generate unique IDs using UUID, optionally combining with file_path if available
        texts_list = list(texts)
        metadatas_list = list(metadatas)
        ids = []
        for idx, metadata in enumerate(metadatas_list):
            # Try to create a more meaningful ID using file_path if available
            file_path = (
                metadata.get("file_path") or 
                metadata.get("file_name") or 
                metadata.get("filename") or 
                metadata.get("source")
            )
            if file_path:
                # Create ID that includes file_path and chunk index for better tracking
                ids.append(f"{file_path}_{idx}_{uuid.uuid4().hex[:8]}")
            else:
                # Fallback to UUID if no file_path available
                ids.append(str(uuid.uuid4()))
        
        self._collection.add(
            ids=ids,
            documents=texts_list,
            metadatas=metadatas_list,
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
            
            # Extract unique documents based on file_path or file_name
            seen_docs = {}
            metadatas = results.get("metadatas", [])
            ids = results.get("ids", [])
            
            for idx, metadata in enumerate(metadatas):
                if not metadata:
                    continue
                
                # Try to get file_path or file_name from metadata
                doc_key = (
                    metadata.get("file_path") or 
                    metadata.get("file_name") or 
                    metadata.get("filename") or
                    metadata.get("source") or
                    f"Document_{ids[idx]}"
                )
                
                # If we haven't seen this document, add it
                if doc_key not in seen_docs:
                    seen_docs[doc_key] = {
                        "name": doc_key,
                        "file_path": metadata.get("file_path") or metadata.get("file_name") or doc_key,
                        "page": metadata.get("page"),
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
        """
        try:
            # Query ChromaDB for each possible metadata field name
            # ChromaDB where clause supports simple equality checks
            all_ids = set()
            all_chunks = {}
            
            # Try each possible metadata field name
            metadata_fields = ["file_path", "file_name", "filename", "source"]
            for field in metadata_fields:
                try:
                    results = self._collection.get(where={field: file_path})
                    if results and results.get("ids"):
                        for idx, chunk_id in enumerate(results.get("ids", [])):
                            if chunk_id not in all_ids:
                                all_ids.add(chunk_id)
                                all_chunks[chunk_id] = {
                                    "id": chunk_id,
                                    "content": results.get("documents", [])[idx] if idx < len(results.get("documents", [])) else "",
                                    "metadata": results.get("metadatas", [])[idx] if idx < len(results.get("metadatas", [])) else {},
                                }
                except Exception:
                    # Continue if this field doesn't exist or query fails
                    continue
            
            return list(all_chunks.values())
        except Exception as e:
            # Return empty list if there's an error
            return []

    def delete_document(self, file_path: str) -> int:
        """
        Delete all chunks for a specific document identified by file_path.
        Returns the number of chunks deleted.
        """
        try:
            # First, get all chunks for this document to count them
            chunks = self.get_document_chunks(file_path)
            if not chunks:
                return 0
            
            chunk_count = len(chunks)
            
            # Delete using metadata filter for each possible field
            # ChromaDB delete supports where clause with simple equality checks
            deleted_count = 0
            metadata_fields = ["file_path", "file_name", "filename", "source"]
            
            for field in metadata_fields:
                try:
                    # Delete chunks matching this field
                    self._collection.delete(where={field: file_path})
                    # Note: ChromaDB delete doesn't return count, so we track by trying each field
                    # Since we already counted chunks above, we'll return that count
                except Exception:
                    # Continue if this field doesn't exist or deletion fails
                    continue
            
            # Return the count we got from get_document_chunks
            # Note: If deletion partially fails, this might not be accurate,
            # but it's the best we can do without ChromaDB returning deletion count
            return chunk_count
        except Exception as e:
            # Return 0 if deletion fails
            return 0


