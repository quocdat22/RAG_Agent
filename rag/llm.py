import logging
from typing import List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings
from rag.prompts import SYSTEM_PROMPT

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

    raise RuntimeError(
        "No chat model provider configured. "
        "Set either Azure OpenAI envs or GITHUB_TOKEN for GitHub Models."
    )


def generate_answer(
    query: str,
    context_chunks: List[str],
    conversation_history: Optional[List[dict]] = None,
) -> str:
    if not context_chunks:
        return "Không tìm thấy trong tài liệu nội bộ."

    client, model_name = _get_client_and_model()

    context_joined = "\n\n---\n\n".join(context_chunks)
    user_content = (
        f"Context:\n{context_joined}\n\n"
        f"Câu hỏi: {query}\n\n"
        "Chỉ trả lời dựa trên context. Nếu context không đủ, trả lời: "
        "'Không tìm thấy trong tài liệu nội bộ.'\n\n"
        "LƯU Ý QUAN TRỌNG:\n"
        "- Nếu context chứa dữ liệu dạng bảng (số liệu, thống kê có nhiều cột và hàng), "
        "hãy trình bày lại dưới dạng markdown table với format:\n"
        "| Tiêu đề cột 1 | Tiêu đề cột 2 | Tiêu đề cột 3 |\n"
        "|----------------|----------------|----------------|\n"
        "| Dữ liệu 1      | Dữ liệu 2      | Dữ liệu 3      |\n"
        "- Giữ nguyên số liệu, không làm tròn trừ khi được yêu cầu.\n"
        "- Nếu có nhiều bảng, trình bày từng bảng một cách rõ ràng."
    )

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

    try:
        resp = client.complete(
            model=model_name,
            messages=messages,
        )

        choice = resp.choices[0]
        if not choice or not choice.message:
            logger.error("No message in response choice")
            return "Lỗi: Không nhận được response từ LLM."

        # Handle different response structures
        content = choice.message.content
        logger.debug(f"Response content type: {type(content)}, content: {content}")
        
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
        
        # Fallback: try to convert to string
        result = str(content) if content else "Lỗi: Response rỗng từ LLM."
        logger.warning(f"Using fallback string conversion: {result}")
        return result
    
    except Exception as e:
        logger.exception("Error calling LLM")
        return f"Lỗi khi gọi LLM: {str(e)}"


