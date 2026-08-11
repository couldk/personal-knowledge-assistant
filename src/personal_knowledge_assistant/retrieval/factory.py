from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.retrieval.service import (
    RetrievalService,
)
from personal_knowledge_assistant.retrieval.tracing import (
    LoggingRetrievalTracer,
    RetrievalTracer,
)


def create_retrieval_service(
    *,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    tracer: RetrievalTracer | None = None,
) -> RetrievalService:
    """使用共享 Provider 创建检索服务。"""

    selected_tracer = tracer if tracer is not None else LoggingRetrievalTracer()

    return RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tracer=selected_tracer,
        embedding_model=settings.embedding_model,
    )
