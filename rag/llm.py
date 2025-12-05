import logging
import time
from typing import List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings
from rag.exceptions import ConfigurationError, LLMError
from rag.logging_utils import log_api_call
from rag.prompts import SYSTEM_PROMPT
from rag.retry_utils import retry_with_backoff, is_retryable_error

logger = logging.getLogger(__name__)


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

    # Handle long context: estimate token count and provide guidance
    context_joined = "\n\n---\n\n".join(context_chunks)
    context_length = len(context_joined)
    
    # Build user prompt with context-aware instructions
    user_content_parts = [
        f"Context (từ {len(context_chunks)} đoạn tài liệu):\n{context_joined}\n\n",
        f"Câu hỏi: {query}\n\n"
    ]
    
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
        
        # Log failed API call
        log_api_call(
            service_name="LLM",
            request_data=request_data,
            duration_ms=duration_ms,
            error=e,
        )
        
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
            raise LLMError(
                f"LLM API call failed: {str(e)}",
                user_message=(
                    "Lỗi khi gọi dịch vụ AI. "
                    "Vui lòng kiểm tra lại câu hỏi hoặc liên hệ quản trị viên nếu vấn đề tiếp tục."
                ),
                details={"error_type": type(e).__name__, "retryable": False}
            ) from e


