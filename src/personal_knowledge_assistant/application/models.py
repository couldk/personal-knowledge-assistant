from dataclasses import dataclass

from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
)
from personal_knowledge_assistant.providers.base import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.retrieval import (
    RetrievalService,
)


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """应用内共享的长生命周期服务。"""

    embedding_provider: EmbeddingProvider
    vector_store: VectorStoreProvider
    indexing_service: DocumentIndexingService
    retrieval_service: RetrievalService
