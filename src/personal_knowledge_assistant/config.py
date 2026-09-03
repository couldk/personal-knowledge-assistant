from functools import lru_cache
from pathlib import Path
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

    # JWT身份认证配置
    auth_enabled: bool = False

    # HS256签名密钥。
    # SecretStr可以避免密钥在日志和repr中明文显示。
    auth_jwt_secret_key: SecretStr | None = None

    auth_jwt_algorithm: Literal["HS256",] = "HS256"

    auth_jwt_issuer: str = Field(
        default=("personal-knowledge-assistant"),
        min_length=1,
        max_length=128,
    )

    auth_jwt_audience: str = Field(
        default=("personal-knowledge-assistant-api"),
        min_length=1,
        max_length=128,
    )

    # 认证关闭时，本地开发和后台任务使用的默认租户。
    auth_local_tenant_id: str = Field(
        default="local",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )

    # 认证关闭时，本地开发和后台任务使用的默认用户。
    auth_local_user_id: str = Field(
        default="local-user",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )

    # 聊天模型配置
    chat_provider: str = "replace-me"
    chat_model: str = "replace-me"
    chat_api_key: SecretStr | None = None

    # 聊天模型API基础地址
    chat_base_url: str = "https://api.deepseek.com"

    # 聊天模型请求超时，单位为秒
    chat_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
    )

    # SDK自动重试次数
    chat_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
    )

    # 单次回答允许生成的最大Token数
    chat_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=32768,
    )

    # 基于证据回答默认使用0，便于稳定测试。
    chat_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    # Embedding模型配置
    embedding_provider: str = "replace-me"
    embedding_model: str = "replace-me"
    embedding_api_key: SecretStr | None = None

    # Embedding API基础地址
    embedding_base_url: str = "https://api.siliconflow.cn/v1"

    # Embedding向量维度
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

    # PostgreSQL连接地址
    database_url: str = "postgresql://pka:pka-local-password@localhost:5432/personal_knowledge"

    database_schema: str = Field(
        default="pka",
        min_length=1,
        max_length=63,
        pattern=r"^[a-z_][a-z0-9_]*$",
    )

    database_connect_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
    )

    database_pool_min_size: int = Field(
        default=1,
        ge=0,
        le=20,
    )

    database_pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    # 上传文件保存目录
    upload_directory: Path = Path("data/uploads")

    # 单个上传文件最大10MiB
    upload_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
    )

    # 文档切块配置，单位为Token
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

    # Top-k检索数量
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    # Agent最大检索次数，包括第一次原始问题检索
    agent_max_retrieval_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
    )

    # 判断检索证据是否足够相关的最低相似度
    agent_min_relevance_score: float = Field(
        default=0.35,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_auth_settings(
        self,
    ) -> Self:
        """验证JWT认证配置。"""

        if self.auth_enabled:
            if self.auth_jwt_secret_key is None:
                raise ValueError("auth_jwt_secret_key is required when authentication is enabled.")

            secret = self.auth_jwt_secret_key.get_secret_value()

            if len(secret) < 32:
                raise ValueError("auth_jwt_secret_key must contain at least 32 characters.")

        # 生产环境必须开启认证，防止接口裸露。
        if self.app_env == "production" and not self.auth_enabled:
            raise ValueError("Authentication must be enabled in production.")

        return self

    @model_validator(mode="after")
    def validate_chunking_settings(
        self,
    ) -> Self:
        """确保Chunk重叠长度小于Chunk总长度。"""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        return self

    @model_validator(mode="after")
    def validate_database_pool_settings(
        self,
    ) -> Self:
        """确保连接池最大值不小于最小值。"""

        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError(
                "database_pool_max_size must be greater than or equal to database_pool_min_size."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的应用配置。"""

    return Settings()
