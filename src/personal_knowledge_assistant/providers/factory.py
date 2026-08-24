from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
    EmbeddingProvider,
)
from personal_knowledge_assistant.providers.deepseek_chat import (
    DeepSeekChatProvider,
)
from personal_knowledge_assistant.providers.siliconflow_embedding import (
    SiliconFlowEmbeddingProvider,
)


def create_chat_provider(
    settings: Settings,
) -> ChatProvider:
    """根据项目配置创建聊天模型 Provider。"""

    provider_name = settings.chat_provider.strip().lower()

    if provider_name != "deepseek":
        raise ValueError(f"Unsupported chat provider: {settings.chat_provider}")

    if settings.chat_api_key is None:
        raise ValueError("Chat API key is required.")

    if not settings.chat_model.strip() or settings.chat_model.strip().lower() == "replace-me":
        raise ValueError("Chat model is required.")

    if not settings.chat_base_url.strip():
        raise ValueError("Chat API base URL is required.")

    return DeepSeekChatProvider(
        api_key=settings.chat_api_key.get_secret_value(),
        model=settings.chat_model.strip(),
        base_url=settings.chat_base_url.strip(),
        timeout_seconds=settings.chat_timeout_seconds,
        max_retries=settings.chat_max_retries,
        max_tokens=settings.chat_max_tokens,
        temperature=settings.chat_temperature,
    )


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    """根据项目配置创建 Embedding Provider。"""

    provider_name = settings.embedding_provider.strip().lower()

    if provider_name != "siliconflow":
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")

    if settings.embedding_api_key is None:
        raise ValueError("Embedding API key is required.")

    if settings.embedding_dimension is None:
        raise ValueError("Embedding dimension is required.")

    return SiliconFlowEmbeddingProvider(
        api_key=settings.embedding_api_key.get_secret_value(),
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )
