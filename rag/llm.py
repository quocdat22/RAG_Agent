import logging
import time
import base64
from typing import List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings
from rag.exceptions import ConfigurationError, LLMError
from rag.logging_utils import log_api_call
from rag.prompts import SYSTEM_PROMPT
from rag.retry_utils import retry_with_backoff, is_retryable_error

logger = logging.getLogger(__name__)

# Token limits for different models
# Rough estimate: 1 token ≈ 4 characters for Vietnamese/English mixed text
# For safety, we'll use a conservative estimate
TOKEN_ESTIMATE_CHARS = 3.5  # characters per token (conservative)
MAX_TOKENS_LIMIT = 8000  # Maximum tokens for request body (including prompt, context, query, history)
RESERVED_TOKENS = 2000  # Reserve tokens for system prompt, user query, instructions, and response buffer
MAX_CONTEXT_TOKENS = MAX_TOKENS_LIMIT - RESERVED_TOKENS  # ~6000 tokens for context


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count from text.
    Uses character count with a conservative ratio.
    """
    if not text:
        return 0
    return int(len(text) / TOKEN_ESTIMATE_CHARS)


def truncate_chunks_to_fit(
    chunks: List[str],
    max_tokens: int,
    query: str = "",
    system_prompt: str = "",
    conversation_history: Optional[List[dict]] = None,
) -> List[str]:
    """
    Truncate chunks to fit within token limit.
    Prioritizes earlier chunks (assumed to be more relevant after reranking).
    
    Args:
        chunks: List of context chunks
        max_tokens: Maximum tokens allowed for context
        query: User query (for estimation)
        system_prompt: System prompt (for estimation)
        conversation_history: Conversation history (for estimation)
    
    Returns:
        Truncated list of chunks that fit within token limit
    """
    # Estimate tokens for non-context parts
    query_tokens = estimate_tokens(query)
    system_tokens = estimate_tokens(system_prompt)
    history_tokens = 0
    if conversation_history:
        for msg in conversation_history:
            history_tokens += estimate_tokens(msg.get("content", ""))
    
    # Reserve tokens for instructions and formatting
    instruction_tokens = 500  # Instructions added in user prompt
    reserved = system_tokens + query_tokens + history_tokens + instruction_tokens
    
    # Available tokens for context
    available_tokens = max_tokens - reserved
    
    if available_tokens <= 0:
        logger.warning(f"Very little space for context: {available_tokens} tokens available")
        return chunks[:1] if chunks else []  # Return at least first chunk
    
    # Try to fit as many chunks as possible
    selected_chunks = []
    total_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        
        # If adding this chunk would exceed limit, truncate it
        if total_tokens + chunk_tokens > available_tokens:
            # Try to truncate this chunk to fit
            remaining_tokens = available_tokens - total_tokens
            if remaining_tokens > 100:  # Only if we have meaningful space
                # Truncate chunk: keep first part
                max_chars = int(remaining_tokens * TOKEN_ESTIMATE_CHARS)
                truncated = chunk[:max_chars]
                if truncated:
                    selected_chunks.append(truncated + "\n\n[... phần còn lại đã được cắt bớt để phù hợp với giới hạn ...]")
                break
            else:
                # Not enough space, stop here
                break
        
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
    
    if not selected_chunks and chunks:
        # If we couldn't fit anything, at least return a truncated first chunk
        max_chars = int(available_tokens * TOKEN_ESTIMATE_CHARS)
        selected_chunks.append(chunks[0][:max_chars] + "\n\n[... đã cắt bớt ...]")
    
    logger.info(
        f"Truncated {len(chunks)} chunks to {len(selected_chunks)} chunks "
        f"(estimated {total_tokens} tokens, limit: {available_tokens})"
    )
    
    return selected_chunks


def _get_client_and_model() -> tuple[ChatCompletionsClient, str]:
    """
    Return ChatCompletionsClient and model name.
    Priority:
    - Azure OpenAI if fully configured
    - Otherwise GitHub Models (GITHUB_TOKEN)
    """
    settings = get_settings()

    # Azure path
    if (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_chat_deployment
    ):
        client = ChatCompletionsClient(
            endpoint=settings.azure_openai_endpoint,
            credential=AzureKeyCredential(settings.azure_openai_api_key),
        )
        return client, settings.azure_openai_chat_deployment

    # GitHub Models path
    if settings.github_token:
        client = ChatCompletionsClient(
            endpoint=settings.github_models_endpoint,
            credential=AzureKeyCredential(settings.github_token),
        )
        return client, settings.github_chat_model

    raise ConfigurationError(
        "No chat model provider configured. "
        "Set either Azure OpenAI envs or GITHUB_TOKEN for GitHub Models.",
        user_message="Cấu hình dịch vụ AI chưa được thiết lập. Vui lòng liên hệ quản trị viên."
    )


@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)
def _call_llm_api(client: ChatCompletionsClient, model_name: str, messages: List[dict]):
    """
    Internal function to call LLM API with retry logic.
    """
    return client.complete(
        model=model_name,
        messages=messages,
    )


def generate_image_description(image_data: bytes, prompt: str = "Describe this chart or image in detail for data analysis purposes.") -> str:
    """
    Generate description for an image using Vision LLM.
    
    Args:
        image_data: Raw image bytes
        prompt: Prompt for the description
        
    Returns:
        Description string
    """
    settings = get_settings()
    if not settings.enable_chart_description:
        return ""
        
    try:
        # Get vision model
        model_name = settings.github_vision_model
        
        # Initialize client (same as chat for now, assuming same endpoint/auth)
        if settings.github_token:
            client = ChatCompletionsClient(
                endpoint=settings.github_models_endpoint,
                credential=AzureKeyCredential(settings.github_token),
            )
        elif settings.azure_openai_endpoint and settings.azure_openai_api_key:
             # Fallback to Azure if configured (might need specific deployment for vision)
             # For now assume user configured vision model as main chat model or we use what's available
             client = ChatCompletionsClient(
                endpoint=settings.azure_openai_endpoint,
                credential=AzureKeyCredential(settings.azure_openai_api_key),
            )
             if settings.azure_openai_chat_deployment:
                 model_name = settings.azure_openai_chat_deployment
        else:
            logger.warning("No AI service configured for vision.")
            return ""

        # Encode image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"
        
        # Prepare message with image
        messages = [
            {
                "role": "system",
                "content": "You are a data analyst helper. Your job is to describe charts and images in documents so that text-only models can understand the data. Be precise with numbers and trends."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
        
        # Call API
        resp = _call_llm_api(client, model_name, messages)
        
        if resp and resp.choices and resp.choices[0].message:
            return resp.choices[0].message.content or ""
            
    except Exception as e:
        logger.warning(f"Failed to generate image description: {e}")
        
    return ""


def generate_answer(
    query: str,
    context_chunks: List[str],
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    Generate answer using LLM with improved error handling and logging.
    
    Raises:
        LLMError: If LLM API call fails after retries
        ConfigurationError: If LLM is not configured
    """
    if not context_chunks:
        return "Không tìm thấy trong tài liệu nội bộ."

    start_time = time.time()
    
    try:
        client, model_name = _get_client_and_model()
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            f"Failed to initialize LLM client: {str(e)}",
            user_message="Không thể khởi tạo dịch vụ AI. Vui lòng kiểm tra cấu hình."
        ) from e

    # Truncate chunks to fit within token limit
    truncated_chunks = truncate_chunks_to_fit(
        context_chunks,
        max_tokens=MAX_TOKENS_LIMIT,
        query=query,
        system_prompt=SYSTEM_PROMPT,
        conversation_history=conversation_history,
    )
    
    if not truncated_chunks:
        logger.warning("No chunks available after truncation")
        return "Không thể xử lý context do quá dài. Vui lòng thử lại với câu hỏi cụ thể hơn."
    
    # Handle long context: estimate token count and provide guidance
    context_joined = "\n\n---\n\n".join(truncated_chunks)
    context_length = len(context_joined)
    original_chunk_count = len(context_chunks)
    final_chunk_count = len(truncated_chunks)
    
    # Build user prompt with context-aware instructions
    if original_chunk_count > final_chunk_count:
        user_content_parts = [
            f"Context (từ {final_chunk_count}/{original_chunk_count} đoạn tài liệu, "
            f"đã được tối ưu để phù hợp với giới hạn):\n{context_joined}\n\n",
        ]
    else:
        user_content_parts = [
            f"Context (từ {final_chunk_count} đoạn tài liệu):\n{context_joined}\n\n",
        ]
    
    user_content_parts.append(f"Câu hỏi: {query}\n\n")
    
    # Add context length guidance if context is very long
    if context_length > 10000:  # ~2500 tokens
        user_content_parts.append(
            "LƯU Ý: Context khá dài. Hãy tập trung vào thông tin liên quan trực tiếp đến câu hỏi. "
            "Nếu cần tóm tắt, hãy giữ lại các số liệu và thông tin quan trọng nhất.\n\n"
        )
    
    # Add specific instructions based on query type
    query_lower = query.lower()
    has_table_keywords = any(kw in query_lower for kw in ['bảng', 'table', 'số liệu', 'thống kê', 'doanh thu', 'chi phí', 'danh sách'])
    has_list_keywords = any(kw in query_lower for kw in ['danh sách', 'list', 'các bước', 'quy trình', 'những gì'])
    
    if has_table_keywords:
        user_content_parts.append(
            "CÂU HỎI LIÊN QUAN ĐẾN BẢNG/SỐ LIỆU:\n"
            "- Nếu context chứa dữ liệu dạng bảng, hãy trình bày lại dưới dạng markdown table\n"
            "- Giữ nguyên số liệu chính xác, không làm tròn\n"
            "- Đặt tiêu đề rõ ràng cho bảng\n\n"
        )
    
    if has_list_keywords:
        user_content_parts.append(
            "CÂU HỎI LIÊN QUAN ĐẾN DANH SÁCH:\n"
            "- Trình bày dưới dạng danh sách có thứ tự (1., 2., 3.) hoặc bullet points (-)\n"
            "- Mỗi item nên ngắn gọn, rõ ràng\n\n"
        )
    
    # Add conflict detection instruction
    user_content_parts.append(
        "XỬ LÝ ĐẶC BIỆT:\n"
        "- Nếu phát hiện thông tin mâu thuẫn trong context, hãy trình bày cả hai và ghi chú về sự khác biệt\n"
        "- Nếu thông tin không đầy đủ, hãy trả lời phần có thể và ghi chú phần còn thiếu\n"
        "- Nếu có nhiều nguồn thông tin, hãy tổng hợp một cách logic\n\n"
    )
    
    user_content_parts.append(
        "Hãy trả lời dựa trên context trên. Nếu context không đủ để trả lời, "
        "hãy trả lời: 'Không tìm thấy trong tài liệu nội bộ.'"
    )
    
    user_content = "".join(user_content_parts)

    # Build messages array with conversation history if provided
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history before the current query
    if conversation_history:
        # Filter to only include user and assistant messages
        for hist_msg in conversation_history:
            if hist_msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": hist_msg["role"],
                    "content": hist_msg["content"]
                })
    
    # Add current query with context
    messages.append({"role": "user", "content": user_content})

    # Prepare request data for logging (sanitize sensitive info)
    request_data = {
        "model": model_name,
        "message_count": len(messages),
        "query_length": len(query),
        "context_chunks_count": len(context_chunks),
        "has_history": bool(conversation_history),
    }

    try:
        # Call LLM API with retry logic
        resp = _call_llm_api(client, model_name, messages)
        duration_ms = (time.time() - start_time) * 1000

        # Log successful API call
        log_api_call(
            service_name="LLM",
            request_data=request_data,
            response_data={
                "has_response": resp is not None,
                "choices_count": len(resp.choices) if resp and hasattr(resp, "choices") else 0,
            },
            duration_ms=duration_ms,
        )

        choice = resp.choices[0]
        if not choice or not choice.message:
            logger.error("No message in response choice")
            raise LLMError(
                "No message in response choice",
                user_message="Không nhận được phản hồi từ dịch vụ AI. Vui lòng thử lại."
            )

        # Handle different response structures
        content = choice.message.content
        logger.debug(f"Response content type: {type(content)}")
        
        # Case 1: content is a string directly
        if isinstance(content, str):
            return content
        
        # Case 2: content is a list of content parts
        if isinstance(content, list):
            text_parts = []
            for part in content:
                # Try different attributes: text, content, etc.
                if hasattr(part, "text"):
                    text_parts.append(part.text)
                elif hasattr(part, "content"):
                    text_parts.append(part.content)
                elif isinstance(part, str):
                    text_parts.append(part)
                elif hasattr(part, "__dict__"):
                    # Try to find any text-like attribute
                    for attr in ["text", "content", "message"]:
                        if hasattr(part, attr):
                            val = getattr(part, attr)
                            if isinstance(val, str):
                                text_parts.append(val)
                                break
            
            if text_parts:
                return "".join(text_parts)
            else:
                logger.warning(f"Could not extract text from content list: {content}")
                raise LLMError(
                    f"Could not extract text from content list: {type(content)}",
                    user_message="Không thể xử lý phản hồi từ dịch vụ AI. Vui lòng thử lại."
                )
        
        # Fallback: try to convert to string
        result = str(content) if content else None
        if not result:
            raise LLMError(
                "Empty response from LLM",
                user_message="Nhận được phản hồi rỗng từ dịch vụ AI. Vui lòng thử lại."
            )
        
        logger.warning(f"Using fallback string conversion: {result[:100]}...")
        return result
    
    except LLMError:
        # Re-raise LLMError as-is
        raise
    except ConfigurationError:
        # Re-raise ConfigurationError as-is
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Check if error is due to token limit
        error_str = str(e).lower()
        is_token_limit_error = (
            "tokens_limit_reached" in error_str or
            "request body too large" in error_str or
            "token limit" in error_str or
            "max size" in error_str
        )
        
        # Log failed API call
        log_api_call(
            service_name="LLM",
            request_data=request_data,
            duration_ms=duration_ms,
            error=e,
        )
        
        # If token limit error and we have chunks, try with fewer chunks
        if is_token_limit_error and len(context_chunks) > 1:
            logger.warning(
                f"Token limit exceeded with {len(context_chunks)} chunks. "
                f"Retrying with fewer chunks..."
            )
            try:
                # Retry with only top 2-3 chunks, more aggressively truncated
                reduced_chunks = truncate_chunks_to_fit(
                    context_chunks[:3],  # Only top 3 chunks
                    max_tokens=MAX_TOKENS_LIMIT - 1000,  # More conservative limit
                    query=query,
                    system_prompt=SYSTEM_PROMPT,
                    conversation_history=conversation_history,
                )
                
                if reduced_chunks:
                    # Rebuild messages with reduced context
                    context_joined = "\n\n---\n\n".join(reduced_chunks)
                    user_content = (
                        f"Context (từ {len(reduced_chunks)} đoạn tài liệu quan trọng nhất, "
                        f"đã được tối ưu):\n{context_joined}\n\n"
                        f"Câu hỏi: {query}\n\n"
                        "Hãy trả lời dựa trên context trên. "
                        "Nếu context không đủ để trả lời, hãy trả lời: 'Không tìm thấy trong tài liệu nội bộ.'"
                    )
                    
                    # Rebuild messages
                    retry_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    if conversation_history:
                        for hist_msg in conversation_history:
                            if hist_msg.get("role") in ("user", "assistant"):
                                retry_messages.append({
                                    "role": hist_msg["role"],
                                    "content": hist_msg["content"]
                                })
                    retry_messages.append({"role": "user", "content": user_content})
                    
                    # Retry with reduced context
                    resp = _call_llm_api(client, model_name, retry_messages)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    choice = resp.choices[0]
                    if choice and choice.message:
                        content = choice.message.content
                        if isinstance(content, str):
                            logger.info("Successfully retried with reduced context")
                            return content
                        elif isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if hasattr(part, "text"):
                                    text_parts.append(part.text)
                                elif hasattr(part, "content"):
                                    text_parts.append(part.content)
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            if text_parts:
                                logger.info("Successfully retried with reduced context")
                                return "".join(text_parts)
            except Exception as retry_error:
                logger.error(f"Retry with reduced context also failed: {retry_error}")
                # Fall through to original error handling
        
        # Determine if error is retryable
        if is_retryable_error(e):
            raise LLMError(
                f"LLM API call failed after retries: {str(e)}",
                user_message=(
                    "Không thể kết nối với dịch vụ AI sau nhiều lần thử. "
                    "Có thể do vấn đề mạng hoặc dịch vụ tạm thời không khả dụng. "
                    "Vui lòng thử lại sau vài phút."
                ),
                details={"error_type": type(e).__name__, "retryable": True}
            ) from e
        else:
            # Provide user-friendly message for token limit errors
            if is_token_limit_error:
                raise LLMError(
                    f"LLM API call failed: {str(e)}",
                    user_message=(
                        "Câu hỏi hoặc tài liệu quá dài để xử lý. "
                        "Vui lòng thử lại với câu hỏi cụ thể hơn hoặc chia nhỏ câu hỏi."
                    ),
                    details={"error_type": type(e).__name__, "retryable": False, "token_limit": True}
                ) from e
            else:
                raise LLMError(
                    f"LLM API call failed: {str(e)}",
                    user_message=(
                        "Lỗi khi gọi dịch vụ AI. "
                        "Vui lòng kiểm tra lại câu hỏi hoặc liên hệ quản trị viên nếu vấn đề tiếp tục."
                    ),
                    details={"error_type": type(e).__name__, "retryable": False}
                ) from e


