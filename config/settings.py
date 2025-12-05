from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


