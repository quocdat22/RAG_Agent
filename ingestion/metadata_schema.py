"""
Metadata schema definition and normalization for consistent metadata handling.

This module provides:
- Standardized metadata schema using Pydantic
- Normalization functions to convert various field names to standard names
- Validation functions to ensure metadata consistency
"""
from typing import Any, Dict, Optional
from pathlib import Path
import logging

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


# Standard metadata field names (kept for backward compatibility)
class MetadataFields:
    """Standard metadata field names."""
    FILE_PATH = "file_path"
    DOCUMENT_TYPE = "document_type"
    PAGE = "page"
    CHUNK_INDEX = "chunk_index"
    
    # Structure fields
    HAS_CODE = "has_code"
    CODE_BLOCK_COUNT = "code_block_count"
    CODE_LANGUAGE = "code_language"
    HAS_TABLE = "has_table"
    TABLE_COUNT = "table_count"
    HAS_HEADERS = "has_headers"
    HEADER_COUNT = "header_count"
    SECTION_HEADERS = "section_headers"
    HAS_LIST = "has_list"
    
    # Version tracking fields
    FILE_HASH = "file_hash"
    FILE_MTIME = "file_mtime"
    DOCUMENT_VERSION = "document_version"


# Legacy field names that should be normalized
LEGACY_FIELD_MAPPING = {
    "file_name": MetadataFields.FILE_PATH,
    "filename": MetadataFields.FILE_PATH,
    "source": MetadataFields.FILE_PATH,
    "page_label": MetadataFields.PAGE,
    "page_number": MetadataFields.PAGE,
}


class DocumentMetadata(BaseModel):
    """
    Pydantic model for document metadata with automatic validation and normalization.
    
    This model ensures consistent metadata structure across the entire system.
    """
    # Required fields
    file_path: str = Field(
        ...,
        description="Path to the source file (required)"
    )
    
    # Optional standard fields
    document_type: Optional[str] = Field(
        None,
        description="Type of document (e.g., 'pdf', 'markdown', 'text')"
    )
    page: Optional[str] = Field(
        None,
        description="Page number or label"
    )
    chunk_index: Optional[int] = Field(
        None,
        description="Index of this chunk in the document",
        ge=0
    )
    
    # Structure fields
    has_code: Optional[bool] = Field(
        None,
        description="Whether the chunk contains code blocks"
    )
    code_block_count: Optional[int] = Field(
        None,
        description="Number of code blocks in the chunk",
        ge=0
    )
    code_language: Optional[str] = Field(
        None,
        description="Programming language if code is present"
    )
    has_table: Optional[bool] = Field(
        None,
        description="Whether the chunk contains tables"
    )
    table_count: Optional[int] = Field(
        None,
        description="Number of tables in the chunk",
        ge=0
    )
    has_headers: Optional[bool] = Field(
        None,
        description="Whether the chunk contains headers"
    )
    header_count: Optional[int] = Field(
        None,
        description="Number of headers in the chunk",
        ge=0
    )
    section_headers: Optional[str] = Field(
        None,
        description="Hierarchy of section headers (e.g., 'Title > Subtitle')"
    )
    has_list: Optional[bool] = Field(
        None,
        description="Whether the chunk contains lists"
    )
    
    # Version tracking fields
    file_hash: Optional[str] = Field(
        None,
        description="SHA256 hash of the source file content"
    )
    file_mtime: Optional[float] = Field(
        None,
        description="File modification time (Unix timestamp)"
    )
    document_version: Optional[int] = Field(
        None,
        description="Document version number (incremental)",
        ge=0
    )
    
    # Allow extra fields for extensibility
    model_config = {
        "extra": "allow",
        "validate_assignment": True,
    }
    
    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Dict[str, Any]:
        """
        Normalize legacy field names to standard names before validation.
        
        This handles fields like 'file_name', 'filename', 'source' -> 'file_path'
        """
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        
        # First, copy all non-legacy fields
        for key, value in data.items():
            if key not in LEGACY_FIELD_MAPPING:
                normalized[key] = value
        
        # Normalize legacy fields
        for legacy_field, standard_field in LEGACY_FIELD_MAPPING.items():
            if legacy_field in data:
                value = data[legacy_field]
                # Only set if standard field doesn't already exist
                if standard_field not in normalized:
                    normalized[standard_field] = value
                elif not normalized[standard_field] and value:
                    # If standard field is empty but legacy has value, use legacy value
                    normalized[standard_field] = value
        
        # Ensure file_path exists (required field)
        if not normalized.get("file_path"):
            # Try to extract from other fields
            file_path = (
                data.get("file_path") or
                data.get("file_name") or
                data.get("filename") or
                data.get("source") or
                "unknown"
            )
            normalized["file_path"] = file_path
        
        # Normalize file_path to be a string
        if normalized.get("file_path"):
            normalized["file_path"] = str(normalized["file_path"])
        
        # Normalize page to be a string if present
        if normalized.get("page") is not None:
            normalized["page"] = str(normalized["page"])
        
        return normalized
    
    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Ensure file_path is not empty."""
        if not v or not v.strip():
            return "unknown"
        return str(v)


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metadata by converting legacy field names to standard names.
    
    Uses Pydantic model for validation and normalization.
    
    Args:
        metadata: Raw metadata dictionary with potentially inconsistent field names
        
    Returns:
        Normalized metadata dictionary with standard field names
    """
    if not metadata:
        # Return minimal valid metadata
        return {"file_path": "unknown"}
    
    try:
        # Use Pydantic model for automatic normalization and validation
        model = DocumentMetadata.model_validate(metadata)
        # Convert back to dict, excluding None values for cleaner output
        result = model.model_dump(exclude_none=False, exclude_unset=False)
        return result
    except Exception as e:
        # Fallback to manual normalization if Pydantic validation fails
        logger.warning(f"Pydantic validation failed, using fallback normalization: {e}")
        
        normalized = {}
        
        # First, copy all non-legacy fields
        for key, value in metadata.items():
            if key not in LEGACY_FIELD_MAPPING:
                normalized[key] = value
        
        # Normalize legacy fields
        for legacy_field, standard_field in LEGACY_FIELD_MAPPING.items():
            if legacy_field in metadata:
                value = metadata[legacy_field]
                # Only set if standard field doesn't already exist
                if standard_field not in normalized:
                    normalized[standard_field] = value
                elif not normalized[standard_field] and value:
                    # If standard field is empty but legacy has value, use legacy value
                    normalized[standard_field] = value
        
        # Ensure file_path exists (required field)
        if not normalized.get(MetadataFields.FILE_PATH):
            # Try to extract from other fields
            file_path = (
                metadata.get("file_path") or
                metadata.get("file_name") or
                metadata.get("filename") or
                metadata.get("source") or
                "unknown"
            )
            normalized[MetadataFields.FILE_PATH] = file_path
        
        # Normalize file_path to be a string
        file_path = normalized.get(MetadataFields.FILE_PATH)
        if file_path:
            normalized[MetadataFields.FILE_PATH] = str(file_path)
        
        return normalized


def validate_metadata(metadata: Dict[str, Any], strict: bool = False) -> tuple[bool, Optional[str]]:
    """
    Validate metadata against schema using Pydantic.
    
    Args:
        metadata: Metadata dictionary to validate
        strict: If True, require all mandatory fields. If False, only validate types.
                (Note: file_path is always required by Pydantic model)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(metadata, dict):
        return False, "Metadata must be a dictionary"
    
    try:
        # Use Pydantic model for validation
        DocumentMetadata.model_validate(metadata)
        return True, None
    except ValidationError as e:
        # Format Pydantic validation errors nicely
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error.get("loc", []))
            msg = error.get("msg", "Validation error")
            errors.append(f"{field}: {msg}")
        error_msg = "; ".join(errors)
        return False, error_msg
    except Exception as e:
        # Handle other exceptions
        return False, f"Validation error: {str(e)}"


def get_file_path(metadata: Dict[str, Any]) -> str:
    """
    Extract file_path from metadata, handling normalization.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        File path as string, or "unknown" if not found
    """
    normalized = normalize_metadata(metadata)
    return normalized.get(MetadataFields.FILE_PATH, "unknown")


def create_metadata(
    file_path: str,
    document_type: Optional[str] = None,
    page: Optional[str] = None,
    chunk_index: Optional[int] = None,
    **extra_fields: Any
) -> Dict[str, Any]:
    """
    Create a standardized metadata dictionary using Pydantic model.
    
    Args:
        file_path: Path to the source file (required)
        document_type: Type of document (e.g., "pdf", "markdown", "text")
        page: Page number or label
        chunk_index: Index of this chunk in the document
        **extra_fields: Additional metadata fields
        
    Returns:
        Normalized metadata dictionary
    """
    # Use Pydantic model for creation and validation
    metadata_dict = {
        "file_path": file_path,
    }
    
    if document_type is not None:
        metadata_dict["document_type"] = document_type
    
    if page is not None:
        metadata_dict["page"] = str(page) if not isinstance(page, str) else page
    
    if chunk_index is not None:
        metadata_dict["chunk_index"] = chunk_index
    
    # Add extra fields
    metadata_dict.update(extra_fields)
    
    # Use Pydantic model for validation and normalization
    model = DocumentMetadata.model_validate(metadata_dict)
    return model.model_dump(exclude_none=False, exclude_unset=False)

