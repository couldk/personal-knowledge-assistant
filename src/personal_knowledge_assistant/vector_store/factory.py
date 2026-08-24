from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import (
    VectorStoreProvider,
)
from personal_knowledge_assistant.vector_store.memory import (
    InMemoryVectorStore,
)


def create_vector_store(
    settings: Settings,
) -> VectorStoreProvider:
    """根据配置创建向量存储。"""

    if settings.embedding_dimension is None:
        raise ValueError("Embedding dimension is required.")

    if settings.vector_store_provider == "memory":
        return InMemoryVectorStore(
            dimension=settings.embedding_dimension,
        )

    if settings.vector_store_provider == "pgvector":
        raise NotImplementedError("PgVectorStore has not been implemented yet.")

    raise ValueError(f"Unsupported vector store provider: {settings.vector_store_provider}")
