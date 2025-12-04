"""
Retry utilities for API calls with exponential backoff.
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional, Type, Tuple

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    check_retryable: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator to retry function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
        check_retryable: If True, only retry errors that pass is_retryable_error check
        on_retry: Optional callback function called on each retry (exception, attempt_number)
    
    Usage:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_api(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if error is retryable (if check_retryable is enabled)
                    if check_retryable and not is_retryable_error(e):
                        logger.debug(
                            f"Function {func.__name__} failed with non-retryable error: "
                            f"{type(e).__name__}: {str(e)}"
                        )
                        raise
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay:.2f}s... Error: {type(e).__name__}: {str(e)}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception:
                            pass  # Don't let retry callback break the retry logic
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.
    
    Common retryable errors:
    - Network errors (connection, timeout)
    - Rate limiting (429)
    - Server errors (5xx)
    - Temporary unavailability
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    # Network-related errors
    network_keywords = [
        "connection", "timeout", "network", "unreachable",
        "refused", "reset", "broken pipe", "ssl"
    ]
    if any(keyword in error_str or keyword in error_type for keyword in network_keywords):
        return True
    
    # HTTP status codes (if exception has status_code attribute)
    if hasattr(exception, "status_code"):
        status = exception.status_code
        # Retry on 429 (rate limit), 5xx (server errors), 408 (timeout)
        if status in (408, 429) or (500 <= status < 600):
            return True
    
    # Rate limiting indicators
    if "rate limit" in error_str or "429" in error_str:
        return True
    
    # Server error indicators
    if "server error" in error_str or "500" in error_str or "503" in error_str:
        return True
    
    return False

