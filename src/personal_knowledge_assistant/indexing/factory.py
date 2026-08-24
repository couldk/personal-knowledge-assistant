from personal_knowledge_assistant.chunking import (
    SentenceChunker,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
)
from personal_knowledge_assistant.indexing.service import (
    DocumentIndexingService,
)
from personal_knowledge_assistant.providers.factory import (
    create_embedding_provider,
)
from personal_knowledge_assistant.vector_store.factory import (
    create_vector_store,
)


def create_indexing_service(
    settings: Settings,
) -> DocumentIndexingService:
    """根据应用配置创建文档索引服务。"""

    chunker = SentenceChunker(
        ChunkingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )

    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)

    return DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
