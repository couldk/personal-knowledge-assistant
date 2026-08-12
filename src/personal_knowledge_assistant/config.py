from functools import lru_cache
from typing import Literal, Self

from pydantic import (
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """个人知识助手的应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PKA_",
        env_ignore_empty=True,
        extra="ignore",
    )

    # 应用运行环境配置
    app_env: Literal[
        "local",
        "test",
        "staging",
        "production",
    ] = "local"

    # 日志输出等级
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    ] = "INFO"

    # 聊天模型配置
    chat_provider: str = "replace-me"
    chat_model: str = "replace-me"
    chat_api_key: SecretStr | None = None

    # 聊天模型 API 基础地址
    chat_base_url: str = "https://api.deepseek.com"

    # 聊天模型请求超时，单位为秒
    chat_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
    )

    # SDK 自动重试次数
    chat_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    # 单次回答允许生成的最大 Token 数
    chat_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=32768,
    )

    # 生成随机性。基于证据回答使用 0，便于稳定测试。
    chat_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    # Embedding 模型配置
    embedding_provider: str = "replace-me"
    embedding_model: str = "replace-me"
    embedding_api_key: SecretStr | None = None

    # Embedding API 基础地址
    embedding_base_url: str = "https://api.siliconflow.cn/v1"

    # Embedding 向量维度
    embedding_dimension: int | None = Field(
        default=None,
        ge=1,
        le=65536,
    )

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
    )
    embedding_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
    )
    embedding_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    # 向量存储配置
    vector_store_provider: Literal[
        "memory",
        "pgvector",
    ] = "pgvector"

    # PostgreSQL连接地址。
    # 当前只是声明配置，尚未建立真实数据库连接。
    database_url: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # 文档切块配置，单位为Token。
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=8192,
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=2048,
    )

    # Top-k检索数量配置
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    @model_validator(mode="after")
    def validate_chunking_settings(self) -> Self:
        """确保Chunk重叠长度小于Chunk总长度。"""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的应用配置。"""

    return Settings()
