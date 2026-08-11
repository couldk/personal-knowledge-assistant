from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import EmbeddingProvider
from personal_knowledge_assistant.providers.siliconflow_embedding import (
    SiliconFlowEmbeddingProvider,
)


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    if settings.embedding_provider != "siliconflow":
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
