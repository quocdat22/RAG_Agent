from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Azure OpenAI (optional, you can skip if using GitHub Models)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_embedding_deployment: Optional[str] = None
    azure_openai_chat_deployment: Optional[str] = None

    # GitHub Models via azure-ai-inference
    github_token: Optional[str] = None  # env: GITHUB_TOKEN
    github_models_endpoint: str = "https://models.github.ai/inference"
    github_embedding_model: str = "openai/text-embedding-3-small"
    github_chat_model: str = "openai/gpt-4.1-mini"

    # Cohere
    cohere_api_key: Optional[str] = None

    # Vector store
    chroma_persist_dir: str = "./data/chroma"

    # Conversation storage
    conversation_db_path: str = "./data/conversations.db"

    # API Authentication
    api_key: Optional[str] = None  # API key for authentication (env: API_KEY)
    api_key_header: str = "X-API-Key"  # Header name for API key

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow reloading from .env file
        env_file_reload=True,
    )


# Global settings instance (can be reloaded)
_settings_instance: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get settings instance. Can reload from .env file if reload=True.
    
    Args:
        reload: If True, reload settings from .env file (useful for API key rotation)
    
    Returns:
        Settings instance
    """
    global _settings_instance
    if reload or _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


