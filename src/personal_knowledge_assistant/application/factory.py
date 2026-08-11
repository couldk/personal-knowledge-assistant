from personal_knowledge_assistant.application.models import (
    ApplicationServices,
)
from personal_knowledge_assistant.chunking import (
    SentenceChunker,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
)
from personal_knowledge_assistant.providers.factory import (
    create_embedding_provider,
)
from personal_knowledge_assistant.retrieval import (
    create_retrieval_service,
)
from personal_knowledge_assistant.vector_store import (
    create_vector_store,
)


def create_application_services(
    settings: Settings,
) -> ApplicationServices:
    """创建并连接应用服务。"""

    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)

    indexing_service = DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=(settings.chunk_overlap),
            )
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    retrieval_service = create_retrieval_service(
        settings=settings,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    return ApplicationServices(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        indexing_service=indexing_service,
        retrieval_service=retrieval_service,
    )
