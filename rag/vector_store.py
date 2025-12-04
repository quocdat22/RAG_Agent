import hashlib
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
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove None values from metadata as ChromaDB doesn't accept them.
        ChromaDB only accepts: Bool, Int, Float, Str, or SparseVector.
        
        Args:
            metadata: Metadata dictionary that may contain None values
            
        Returns:
            Cleaned metadata dictionary without None values
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                cleaned[key] = value
        return cleaned
    
    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """
        Generate deterministic chunk ID based on file_path and chunk_index.
        Uses hash of file_path to avoid special character issues.
        
        Args:
            file_path: Path to the source file
            chunk_index: Index of the chunk in the document
            
        Returns:
            Deterministic chunk ID
        """
        # Hash file_path to create a safe identifier (first 12 chars of SHA256)
        file_path_hash = hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:12]
        return f"{file_path_hash}_{chunk_index}"

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
            # Remove None values as ChromaDB doesn't accept them
            cleaned_metadata = self._clean_metadata(normalized_metadata)
            normalized_metadatas.append(cleaned_metadata)
            
            # Create deterministic ID using normalized file_path and chunk_index
            file_path = cleaned_metadata.get(MetadataFields.FILE_PATH, "unknown")
            # Ensure chunk_index is not None - use idx if missing or None
            chunk_index = cleaned_metadata.get(MetadataFields.CHUNK_INDEX)
            if chunk_index is None:
                chunk_index = idx
                # Add chunk_index to cleaned metadata
                cleaned_metadata[MetadataFields.CHUNK_INDEX] = chunk_index
            ids.append(self._generate_chunk_id(file_path, chunk_index))
        
        self._collection.add(
            ids=ids,
            documents=texts_list,
            metadatas=normalized_metadatas,
            embeddings=list(embeddings),
        )
    
    def upsert_documents(
        self,
        texts: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
        embeddings: Iterable[List[float]],
    ) -> Dict[str, int]:
        """
        Upsert documents with version control.
        Checks if documents exist and compares versions.
        If document changed, deletes old chunks and adds new ones.
        If document unchanged, skips ingestion.
        
        Args:
            texts: List of chunk texts
            metadatas: List of chunk metadata dictionaries
            embeddings: List of embeddings
            
        Returns:
            Dictionary with statistics: updated_count, skipped_count, new_count
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Convert to lists
        texts_list = list(texts)
        metadatas_list = list(metadatas)
        embeddings_list = list(embeddings)
        
        if len(texts_list) != len(metadatas_list) or len(texts_list) != len(embeddings_list):
            raise ValueError("texts, metadatas, and embeddings must have the same length")
        
        # Group chunks by document (file_path)
        document_chunks = {}
        for idx, (text, metadata, embedding) in enumerate(zip(texts_list, metadatas_list, embeddings_list)):
            normalized_metadata = normalize_metadata(metadata)
            file_path = normalized_metadata.get(MetadataFields.FILE_PATH, "unknown")
            
            if file_path not in document_chunks:
                document_chunks[file_path] = {
                    "texts": [],
                    "metadatas": [],
                    "embeddings": [],
                    "indices": []
                }
            
            document_chunks[file_path]["texts"].append(text)
            document_chunks[file_path]["metadatas"].append(normalized_metadata)
            document_chunks[file_path]["embeddings"].append(embedding)
            document_chunks[file_path]["indices"].append(idx)
        
        # Statistics
        stats = {
            "updated_count": 0,
            "skipped_count": 0,
            "new_count": 0
        }
        
        # Process each document
        texts_to_add = []
        metadatas_to_add = []
        embeddings_to_add = []
        ids_to_add = []
        
        for file_path, doc_data in document_chunks.items():
            if file_path == "unknown":
                # Unknown file path - always add
                for idx, (text, metadata, embedding) in enumerate(zip(
                    doc_data["texts"],
                    doc_data["metadatas"],
                    doc_data["embeddings"]
                )):
                    # Ensure chunk_index is not None - use idx if missing or None
                    chunk_index = metadata.get(MetadataFields.CHUNK_INDEX)
                    if chunk_index is None:
                        chunk_index = idx
                        # Update metadata to include chunk_index
                        metadata[MetadataFields.CHUNK_INDEX] = chunk_index
                    
                    # Clean metadata to remove None values (ChromaDB requirement)
                    cleaned_metadata = self._clean_metadata(metadata)
                    
                    chunk_id = self._generate_chunk_id(file_path, chunk_index)
                    ids_to_add.append(chunk_id)
                    texts_to_add.append(text)
                    metadatas_to_add.append(cleaned_metadata)
                    embeddings_to_add.append(embedding)
                stats["new_count"] += len(doc_data["texts"])
                continue
            
            # Get existing document version
            existing_version = self.get_document_version(file_path)
            new_hash = doc_data["metadatas"][0].get(MetadataFields.FILE_HASH)
            
            # Determine if document should be re-ingested
            should_reingest = True
            if existing_version:
                existing_hash = existing_version.get("file_hash")
                if existing_hash and new_hash and existing_hash == new_hash:
                    # Document unchanged - skip
                    should_reingest = False
                    stats["skipped_count"] += len(doc_data["texts"])
                    logger.info(f"Document {file_path} unchanged (hash: {new_hash[:8]}...), skipping re-ingestion")
            
            if should_reingest:
                # Document changed or new - delete old chunks and add new ones
                if existing_version:
                    # Delete old chunks - this will raise if deletion fails
                    try:
                        deleted_count = self.delete_document(file_path)
                        logger.info(f"Deleted {deleted_count} old chunks for {file_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to delete old chunks for {file_path}: {e}. "
                            "Skipping re-ingestion to prevent duplicate chunks."
                        )
                        # Skip this document to prevent duplicate ID errors
                        stats["skipped_count"] += len(doc_data["texts"])
                        continue
                    
                    stats["updated_count"] += len(doc_data["texts"])
                    
                    # Increment version
                    current_version = existing_version.get("document_version", 0)
                    new_version = current_version + 1
                else:
                    # New document
                    new_version = 1
                    stats["new_count"] += len(doc_data["texts"])
                
                # Add version to all chunks
                for metadata in doc_data["metadatas"]:
                    metadata[MetadataFields.DOCUMENT_VERSION] = new_version
                
                # Prepare chunks for addition
                for idx, (text, metadata, embedding) in enumerate(zip(
                    doc_data["texts"],
                    doc_data["metadatas"],
                    doc_data["embeddings"]
                )):
                    # Ensure chunk_index is not None - use idx if missing or None
                    chunk_index = metadata.get(MetadataFields.CHUNK_INDEX)
                    if chunk_index is None:
                        chunk_index = idx
                        # Update metadata to include chunk_index
                        metadata[MetadataFields.CHUNK_INDEX] = chunk_index
                    
                    # Clean metadata to remove None values (ChromaDB requirement)
                    cleaned_metadata = self._clean_metadata(metadata)
                    
                    chunk_id = self._generate_chunk_id(file_path, chunk_index)
                    ids_to_add.append(chunk_id)
                    texts_to_add.append(text)
                    metadatas_to_add.append(cleaned_metadata)
                    embeddings_to_add.append(embedding)
        
        # Add all chunks in one batch
        if ids_to_add:
            # Check for duplicate IDs before adding
            seen_ids = set()
            duplicate_ids = []
            for i, chunk_id in enumerate(ids_to_add):
                if chunk_id in seen_ids:
                    duplicate_ids.append((i, chunk_id))
                seen_ids.add(chunk_id)
            
            if duplicate_ids:
                # Log detailed information about duplicates
                logger.error(f"Found {len(duplicate_ids)} duplicate chunk IDs:")
                for idx, dup_id in duplicate_ids:
                    file_path = metadatas_to_add[idx].get(MetadataFields.FILE_PATH, "unknown")
                    chunk_index = metadatas_to_add[idx].get(MetadataFields.CHUNK_INDEX, "None")
                    logger.error(f"  Index {idx}: ID={dup_id}, file_path={file_path}, chunk_index={chunk_index}")
                raise ValueError(f"Duplicate chunk IDs found: {[dup_id for _, dup_id in duplicate_ids]}")
            
            try:
                self._collection.add(
                    ids=ids_to_add,
                    documents=texts_to_add,
                    metadatas=metadatas_to_add,
                    embeddings=embeddings_to_add,
                )
                logger.info(f"Upserted {len(ids_to_add)} chunks: {stats['updated_count']} updated, {stats['new_count']} new, {stats['skipped_count']} skipped")
            except Exception as e:
                logger.error(f"Error during upsert: {e}", exc_info=True)
                raise
        
        return stats

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

    def get_document_version(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get version metadata for an existing document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with file_hash, file_mtime, and document_version if document exists,
            None otherwise
        """
        try:
            chunks = self.get_document_chunks(file_path)
            if not chunks:
                return None
            
            # Get version metadata from the first chunk (all chunks should have same version)
            first_chunk = chunks[0]
            metadata = first_chunk.get("metadata", {})
            
            version_info = {}
            if MetadataFields.FILE_HASH in metadata:
                version_info["file_hash"] = metadata[MetadataFields.FILE_HASH]
            if MetadataFields.FILE_MTIME in metadata:
                version_info["file_mtime"] = metadata[MetadataFields.FILE_MTIME]
            if MetadataFields.DOCUMENT_VERSION in metadata:
                version_info["document_version"] = metadata[MetadataFields.DOCUMENT_VERSION]
            else:
                # Backward compatibility: treat missing version as 0
                version_info["document_version"] = 0
            
            return version_info if version_info else None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error getting document version for {file_path}: {e}")
            return None

    def delete_document(self, file_path: str) -> int:
        """
        Delete all chunks for a specific document identified by file_path.
        Returns the number of chunks deleted.
        
        Uses normalized file_path field for consistent deletion.
        
        Raises:
            Exception: If deletion fails, to prevent false success reporting
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # First, get all chunks for this document to count them
            chunks = self.get_document_chunks(file_path)
            if not chunks:
                return 0
            
            chunk_count = len(chunks)
            
            # Delete using normalized file_path field
            try:
                self._collection.delete(where={MetadataFields.FILE_PATH: file_path})
                
                # Verify deletion succeeded by checking if chunks still exist
                remaining_chunks = self.get_document_chunks(file_path)
                if remaining_chunks:
                    # Deletion failed - chunks still exist
                    logger.error(
                        f"Deletion verification failed: {len(remaining_chunks)} chunks still exist "
                        f"for {file_path} after delete operation"
                    )
                    raise RuntimeError(
                        f"Failed to delete chunks for {file_path}: "
                        f"{len(remaining_chunks)} chunks still exist"
                    )
                
                logger.info(f"Successfully deleted {chunk_count} chunks for {file_path}")
                return chunk_count
                
            except Exception as e:
                # Log error and re-raise to prevent false success reporting
                logger.error(f"Error deleting document chunks for {file_path}: {e}", exc_info=True)
                raise RuntimeError(
                    f"Failed to delete chunks for {file_path}: {str(e)}"
                ) from e
                
        except RuntimeError:
            # Re-raise RuntimeError (deletion failure) as-is
            raise
        except Exception as e:
            # Log unexpected errors and return 0
            logger.error(f"Unexpected error in delete_document for {file_path}: {e}", exc_info=True)
            return 0


