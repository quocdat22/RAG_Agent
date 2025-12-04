import logging
import time
from typing import Iterable, List

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings
from rag.exceptions import ConfigurationError, EmbeddingError
from rag.logging_utils import log_api_call
from rag.retry_utils import retry_with_backoff, is_retryable_error

logger = logging.getLogger(__name__)


def _get_client_and_model() -> tuple[EmbeddingsClient, str]:
    """
    Return an EmbeddingsClient and model name.
    Priority:
    - Azure OpenAI if fully configured
    - Otherwise GitHub Models (GITHUB_TOKEN)
    """
    settings = get_settings()

    # Azure path
    if (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_embedding_deployment
    ):
        client = EmbeddingsClient(
            endpoint=settings.azure_openai_endpoint,
            credential=AzureKeyCredential(settings.azure_openai_api_key),
        )
        return client, settings.azure_openai_embedding_deployment

    # GitHub Models path (your sample)
    if settings.github_token:
        client = EmbeddingsClient(
            endpoint=settings.github_models_endpoint,
            credential=AzureKeyCredential(settings.github_token),
        )
        return client, settings.github_embedding_model

    raise ConfigurationError(
        "No embedding provider configured. "
        "Set either Azure OpenAI envs or GITHUB_TOKEN for GitHub Models.",
        user_message="Cấu hình dịch vụ embedding chưa được thiết lập. Vui lòng liên hệ quản trị viên."
    )


@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)
def _call_embedding_api(client: EmbeddingsClient, model_name: str, texts: List[str]):
    """
    Internal function to call embedding API with retry logic.
    """
    return client.embed(
        input=texts,
        model=model_name,
    )


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """
    Embed a list of texts using configured provider (Azure or GitHub Models).
    
    Raises:
        EmbeddingError: If embedding API call fails after retries
        ConfigurationError: If embedding service is not configured
    """
    texts = list(texts)
    if not texts:
        return []

    start_time = time.time()
    
    try:
        client, model_name = _get_client_and_model()
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            f"Failed to initialize embedding client: {str(e)}",
            user_message="Không thể khởi tạo dịch vụ embedding. Vui lòng kiểm tra cấu hình."
        ) from e

    # Prepare request data for logging
    request_data = {
        "model": model_name,
        "text_count": len(texts),
        "total_chars": sum(len(text) for text in texts),
    }

    try:
        # Call embedding API with retry logic
        resp = _call_embedding_api(client, model_name, texts)
        duration_ms = (time.time() - start_time) * 1000

        # Log successful API call
        log_api_call(
            service_name="Embeddings",
            request_data=request_data,
            response_data={
                "has_response": resp is not None,
                "embeddings_count": len(resp.data) if resp and hasattr(resp, "data") else 0,
            },
            duration_ms=duration_ms,
        )

        if not resp or not hasattr(resp, "data") or not resp.data:
            raise EmbeddingError(
                "Empty response from embedding API",
                user_message="Không nhận được phản hồi từ dịch vụ embedding. Vui lòng thử lại."
            )

        embeddings = [d.embedding for d in resp.data]
        
        if len(embeddings) != len(texts):
            logger.warning(
                f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        
        return embeddings
    
    except EmbeddingError:
        # Re-raise EmbeddingError as-is
        raise
    except ConfigurationError:
        # Re-raise ConfigurationError as-is
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Log failed API call
        log_api_call(
            service_name="Embeddings",
            request_data=request_data,
            duration_ms=duration_ms,
            error=e,
        )
        
        # Determine if error is retryable
        if is_retryable_error(e):
            raise EmbeddingError(
                f"Embedding API call failed after retries: {str(e)}",
                user_message=(
                    "Không thể kết nối với dịch vụ embedding sau nhiều lần thử. "
                    "Có thể do vấn đề mạng hoặc dịch vụ tạm thời không khả dụng. "
                    "Vui lòng thử lại sau vài phút."
                ),
                details={"error_type": type(e).__name__, "retryable": True}
            ) from e
        else:
            raise EmbeddingError(
                f"Embedding API call failed: {str(e)}",
                user_message=(
                    "Lỗi khi gọi dịch vụ embedding. "
                    "Vui lòng thử lại hoặc liên hệ quản trị viên nếu vấn đề tiếp tục."
                ),
                details={"error_type": type(e).__name__, "retryable": False}
            ) from e


