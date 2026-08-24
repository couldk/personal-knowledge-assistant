import asyncio
from datetime import UTC, datetime
from hashlib import sha256

from langchain_core.tracers.langchain import (
    wait_for_all_tracers,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentRequest,
)
from personal_knowledge_assistant.application import (
    create_application_services,
)
from personal_knowledge_assistant.config import (
    Settings,
)
from personal_knowledge_assistant.domain import (
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    ImportResult,
    ImportStatus,
    LoadedDocument,
)


async def main() -> None:
    """使用真实模型和内存向量库验证知识 Agent。"""

    knowledge_text = (
        "RAG 是检索增强生成。"
        "系统首先把用户问题转换成向量，"
        "然后从知识库中检索相关文档片段，"
        "最后要求语言模型依据检索证据生成回答。"
        "这种方式可以减少模型幻觉，并且允许回答携带引用。"
    )

    settings = Settings().model_copy(
        update={
            "vector_store_provider": "memory",
        }
    )

    services = create_application_services(
        settings,
    )

    document = LoadedDocument(
        document_id="memory://agent-verification",
        content_hash=sha256(knowledge_text.encode("utf-8")).hexdigest(),
        metadata=DocumentMetadata(
            source_path=("memory://agent-verification"),
            file_name=("agent-verification.txt"),
            document_type=DocumentType.TEXT,
            file_size=len(knowledge_text.encode("utf-8")),
            modified_at=datetime.now(UTC),
        ),
        parts=[
            DocumentPart(
                part_index=0,
                text=knowledge_text,
            )
        ],
    )

    await services.indexing_service.index(
        ImportResult(
            status=ImportStatus.CREATED,
            document=document,
        )
    )

    thread_id = "day6-real-verification"

    result = await services.agent_service.run(
        KnowledgeAgentRequest(
            question="RAG 的工作流程是什么？",
            top_k=3,
            max_retrieval_attempts=2,
            min_relevance_score=(settings.agent_min_relevance_score),
        ),
        thread_id=thread_id,
    )

    history = await services.agent_service.get_question_history(
        thread_id=thread_id,
    )

    print("Knowledge Agent verification: PASSED")
    print("Outcome:", result.outcome.value)
    print("Answer:", result.answer.answer)
    print(
        "Refused:",
        result.answer.refused,
    )
    print(
        "Confidence:",
        result.answer.confidence,
    )
    print(
        "Retrieval attempts:",
        result.retrieval_attempts,
    )
    print(
        "Final retrieval query:",
        result.final_retrieval_query,
    )
    print(
        "Citation IDs:",
        [citation.chunk_id for citation in result.answer.citations],
    )
    print(
        "Evidence files:",
        [evidence.file_name for evidence in result.evidence],
    )
    print(
        "Question history:",
        history,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        wait_for_all_tracers()
