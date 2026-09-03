from personal_knowledge_assistant.auth.context import (
    get_current_tenant_id,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import (
    VectorStoreProvider,
)
from personal_knowledge_assistant.vector_store.memory import (
    InMemoryVectorStore,
)
from personal_knowledge_assistant.vector_store.postgres import (
    PgVectorStore,
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
        return PgVectorStore(
            database_url=settings.database_url,
            schema=settings.database_schema,
            dimension=settings.embedding_dimension,
            pool_min_size=(settings.database_pool_min_size),
            pool_max_size=(settings.database_pool_max_size),
            connect_timeout_seconds=(settings.database_connect_timeout_seconds),
            tenant_id=settings.auth_local_tenant_id,
            tenant_id_provider=lambda: get_current_tenant_id(settings.auth_local_tenant_id),
        )

    raise ValueError(f"Unsupported vector store provider: {settings.vector_store_provider}")
