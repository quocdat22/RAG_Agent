"""
Custom exceptions for RAG Agent with user-friendly error messages.
"""


class RAGAgentException(Exception):
    """Base exception for RAG Agent."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        """
        Args:
            message: Technical error message for logging
            user_message: User-friendly error message
            details: Additional error details
        """
        super().__init__(message)
        self.user_message = user_message or message
        self.details = details or {}


class LLMError(RAGAgentException):
    """Exception raised when LLM API call fails."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Xin lỗi, hệ thống không thể kết nối với dịch vụ AI. "
            "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )


class EmbeddingError(RAGAgentException):
    """Exception raised when embedding API call fails."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Xin lỗi, hệ thống không thể xử lý embedding. "
            "Vui lòng thử lại sau hoặc liên hệ quản trị viên."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )


class RetrievalError(RAGAgentException):
    """Exception raised when retrieval fails."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Xin lỗi, hệ thống không thể tìm kiếm thông tin trong tài liệu. "
            "Vui lòng thử lại với câu hỏi khác."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )


class ConfigurationError(RAGAgentException):
    """Exception raised when configuration is invalid."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Cấu hình hệ thống không hợp lệ. "
            "Vui lòng liên hệ quản trị viên để kiểm tra cài đặt."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )


class StorageError(RAGAgentException):
    """Exception raised when storage operations fail."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Xin lỗi, hệ thống không thể lưu trữ dữ liệu. "
            "Vui lòng thử lại sau."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )


class IngestionError(RAGAgentException):
    """Exception raised when ingestion fails."""
    
    def __init__(self, message: str, user_message: str = None, details: dict = None):
        default_user_message = (
            "Xin lỗi, hệ thống không thể xử lý tài liệu. "
            "Vui lòng kiểm tra định dạng file và thử lại."
        )
        super().__init__(
            message,
            user_message or default_user_message,
            details
        )

