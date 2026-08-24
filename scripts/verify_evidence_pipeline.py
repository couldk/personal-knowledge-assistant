import asyncio
from datetime import UTC, datetime
from hashlib import sha256

from personal_knowledge_assistant.application import (
    create_application_services,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    ImportResult,
    ImportStatus,
    LoadedDocument,
    RetrievalQuery,
)


async def main() -> None:
    text = (
        "RAG 是检索增强生成。它首先从知识库检索相关文档片段，然后让语言模型依据这些证据生成回答。"
    )

    settings = Settings().model_copy(
        update={
            "vector_store_provider": "memory",
        }
    )

    services = create_application_services(settings)

    document = LoadedDocument(
        document_id="memory://rag-verification",
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        metadata=DocumentMetadata(
            source_path="memory://rag-verification",
            file_name="rag-verification.txt",
            document_type=DocumentType.TEXT,
            file_size=len(text.encode("utf-8")),
            modified_at=datetime.now(UTC),
        ),
        parts=[
            DocumentPart(
                part_index=0,
                text=text,
            )
        ],
    )

    await services.indexing_service.index(
        ImportResult(
            status=ImportStatus.CREATED,
            document=document,
        )
    )

    answer = await services.query_service.query(
        RetrievalQuery(
            query="RAG 的工作流程是什么？",
            top_k=3,
        )
    )

    print("Evidence pipeline verification: PASSED")
    print("Answer:", answer.answer)
    print("Confidence:", answer.confidence)
    print("Refused:", answer.refused)
    print(
        "Citation IDs:",
        [citation.chunk_id for citation in answer.citations],
    )


if __name__ == "__main__":
    asyncio.run(main())
