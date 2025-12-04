from typing import Iterable, List

from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential

from config.settings import get_settings


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

    raise RuntimeError(
        "No embedding provider configured. "
        "Set either Azure OpenAI envs or GITHUB_TOKEN for GitHub Models."
    )


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Embed a list of texts using configured provider (Azure or GitHub Models)."""
    texts = list(texts)
    if not texts:
        return []

    client, model_name = _get_client_and_model()

    resp = client.embed(
        input=texts,
        model=model_name,
    )

    return [d.embedding for d in resp.data]


