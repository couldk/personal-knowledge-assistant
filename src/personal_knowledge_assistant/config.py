from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PKA_",
        env_ignore_empty=True,
        extra="ignore",
    )

    # 应用环境配置
    app_env: Literal["local", "test", "staging", "production"] = "local"
    # 日志级别配置
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # 聊天模型配置
    chat_provider: str = "replace-me"
    chat_model: str = "replace-me"
    chat_api_key: SecretStr | None = None

    # Embedding 配置
    embedding_provider: str = "replace-me"
    embedding_model: str = "replace-me"
    embedding_api_key: SecretStr | None = None

    # 向量数据库配置
    vector_store_provider: str = "pgvector"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # 检索数量配置
    retrieval_top_k: int = Field(default=5, ge=1, le=20)


@lru_cache
def get_settings() -> Settings:
    return Settings()
