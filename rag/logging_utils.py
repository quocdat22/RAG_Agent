"""
Logging utilities for request/response logging and structured logging.
"""

import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def log_api_call(
    service_name: str,
    request_data: Optional[Dict[str, Any]] = None,
    response_data: Optional[Any] = None,
    duration_ms: Optional[float] = None,
    error: Optional[Exception] = None,
    level: int = logging.INFO,
):
    """
    Log API call with request/response information.
    
    Args:
        service_name: Name of the service (e.g., "LLM", "Embeddings")
        request_data: Request data to log (will be sanitized)
        response_data: Response data to log (will be truncated if too long)
        duration_ms: Duration in milliseconds
        error: Exception if call failed
        level: Logging level
    """
    log_data = {
        "service": service_name,
        "timestamp": time.time(),
    }
    
    if request_data:
        # Sanitize request data (remove sensitive info, truncate long strings)
        sanitized_request = _sanitize_data(request_data)
        log_data["request"] = sanitized_request
    
    if error:
        log_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        level = logging.ERROR
    elif response_data is not None:
        # Truncate response if too long
        sanitized_response = _sanitize_data(response_data, max_length=500)
        log_data["response"] = sanitized_response
    
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)
    
    log_message = json.dumps(log_data, ensure_ascii=False, indent=2)
    logger.log(level, f"API Call:\n{log_message}")


def _sanitize_data(data: Any, max_length: int = 1000) -> Any:
    """
    Sanitize data for logging:
    - Remove sensitive fields (api_key, token, password, etc.)
    - Truncate long strings
    - Convert objects to dicts if possible
    """
    if isinstance(data, str):
        if len(data) > max_length:
            return data[:max_length] + f"... (truncated, {len(data)} chars)"
        return data
    
    if isinstance(data, dict):
        sanitized = {}
        sensitive_keys = {"api_key", "token", "password", "credential", "key"}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = _sanitize_data(value, max_length)
        return sanitized
    
    if isinstance(data, (list, tuple)):
        return [_sanitize_data(item, max_length) for item in data[:10]]  # Limit to 10 items
    
    if hasattr(data, "__dict__"):
        return _sanitize_data(data.__dict__, max_length)
    
    return str(data)[:max_length] if len(str(data)) > max_length else data


def log_function_call(
    func_name: str,
    args: tuple = (),
    kwargs: dict = None,
    result: Any = None,
    duration_ms: Optional[float] = None,
    error: Optional[Exception] = None,
    level: int = logging.DEBUG,
):
    """
    Log function call with parameters and result.
    
    Args:
        func_name: Name of the function
        args: Function arguments
        kwargs: Function keyword arguments
        result: Function result
        duration_ms: Duration in milliseconds
        error: Exception if call failed
        level: Logging level
    """
    log_data = {
        "function": func_name,
        "timestamp": time.time(),
    }
    
    if args:
        log_data["args"] = _sanitize_data(args)
    
    if kwargs:
        log_data["kwargs"] = _sanitize_data(kwargs)
    
    if error:
        log_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        level = logging.ERROR
    elif result is not None:
        log_data["result"] = _sanitize_data(result, max_length=200)
    
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)
    
    log_message = json.dumps(log_data, ensure_ascii=False, indent=2)
    logger.log(level, f"Function Call:\n{log_message}")


def log_api_decorator(service_name: str):
    """
    Decorator to automatically log API calls.
    
    Usage:
        @log_api_decorator("LLM")
        def call_llm(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            result = None
            
            # Prepare request data
            request_data = {}
            if args:
                request_data["args"] = args
            if kwargs:
                request_data["kwargs"] = kwargs
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful call
                log_api_call(
                    service_name=service_name,
                    request_data=request_data,
                    response_data=result,
                    duration_ms=duration_ms,
                )
                
                return result
            except Exception as e:
                error = e
                duration_ms = (time.time() - start_time) * 1000
                
                # Log failed call
                log_api_call(
                    service_name=service_name,
                    request_data=request_data,
                    duration_ms=duration_ms,
                    error=error,
                )
                
                raise
        
        return wrapper
    return decorator

