from dataclasses import dataclass

from personal_knowledge_assistant.answering import (
    AnsweringService,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
)
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.querying import (
    KnowledgeQueryService,
)
from personal_knowledge_assistant.retrieval import (
    RetrievalService,
)


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """应用内共享的长生命周期服务。"""

    chat_provider: ChatProvider
    embedding_provider: EmbeddingProvider
    vector_store: VectorStoreProvider
    indexing_service: DocumentIndexingService
    retrieval_service: RetrievalService
    answering_service: AnsweringService
    query_service: KnowledgeQueryService
