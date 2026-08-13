from personal_knowledge_assistant.answering import (
    create_answering_service,
)
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
    create_chat_provider,
    create_embedding_provider,
)
from personal_knowledge_assistant.querying import (
    create_query_service,
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

    chat_provider = create_chat_provider(settings)
    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings)

    indexing_service = DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
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

    answering_service = create_answering_service(
        settings=settings,
        chat_provider=chat_provider,
    )

    query_service = create_query_service(
        retriever=retrieval_service,
        answer_generator=answering_service,
    )

    return ApplicationServices(
        chat_provider=chat_provider,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        indexing_service=indexing_service,
        retrieval_service=retrieval_service,
        answering_service=answering_service,
        query_service=query_service,
    )
