from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

import personal_knowledge_assistant.application.factory as application_factory
from personal_knowledge_assistant.answering import (
    AnsweringTraceStatus,
    InMemoryAnsweringTracer,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    ChatMessage,
    ChatResponse,
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    ImportResult,
    ImportStatus,
    LoadedDocument,
    RetrievalQuery,
)
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
    RetrievalTraceStatus,
)
from personal_knowledge_assistant.vector_store import (
    InMemoryVectorStore,
)


class SharedChatProvider:
    """测试用的共享聊天模型 Provider。"""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.json_mode = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        self.messages = list(messages)
        self.json_mode = json_mode

        return ChatResponse(
            content=('{"answer":"测试回答","citations":[],"confidence":0.0,"refused":true}'),
            model="test-chat-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


class SharedEmbeddingProvider:
    """测试用的共享 Embedding Provider。"""

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        return self._embed(query)

    @staticmethod
    def _embed(
        text: str,
    ) -> list[float]:
        if "vector" in text.lower():
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]


def _make_document() -> LoadedDocument:
    """创建用于索引和检索测试的文档。"""

    text = "Vector retrieval uses shared services."

    return LoadedDocument(
        document_id=("file:///documents/shared.txt"),
        content_hash="a" * 64,
        metadata=DocumentMetadata(
            source_path=("D:/documents/shared.txt"),
            file_name="shared.txt",
            document_type=DocumentType.TEXT,
            file_size=len(text.encode("utf-8")),
            modified_at=datetime(
                2026,
                8,
                11,
                10,
                0,
                tzinfo=UTC,
            ),
        ),
        parts=[
            DocumentPart(
                part_index=0,
                text=text,
            )
        ],
    )


@pytest.mark.asyncio
async def test_application_services_share_dependencies_and_tracers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_provider = SharedChatProvider()
    embedding_provider = SharedEmbeddingProvider()
    vector_store = InMemoryVectorStore(
        dimension=3,
    )

    retrieval_tracer = InMemoryRetrievalTracer()
    answering_tracer = InMemoryAnsweringTracer()

    def fake_create_chat_provider(
        settings: Settings,
    ) -> ChatProvider:
        del settings
        return chat_provider

    def fake_create_embedding_provider(
        settings: Settings,
    ) -> EmbeddingProvider:
        del settings
        return embedding_provider

    def fake_create_vector_store(
        settings: Settings,
    ) -> VectorStoreProvider:
        del settings
        return vector_store

    monkeypatch.setattr(
        application_factory,
        "create_chat_provider",
        fake_create_chat_provider,
    )
    monkeypatch.setattr(
        application_factory,
        "create_embedding_provider",
        fake_create_embedding_provider,
    )
    monkeypatch.setattr(
        application_factory,
        "create_vector_store",
        fake_create_vector_store,
    )

    settings = Settings.model_validate(
        {
            "chat_provider": "deepseek",
            "chat_model": "test-chat-model",
            "chat_api_key": "test-chat-key",
            "chat_base_url": ("https://api.deepseek.com"),
            "embedding_provider": "siliconflow",
            "embedding_model": ("test-embedding-model"),
            "embedding_api_key": ("test-embedding-key"),
            "embedding_dimension": 3,
            "vector_store_provider": "memory",
            "chunk_size": 64,
            "chunk_overlap": 8,
        }
    )

    services = application_factory.create_application_services(
        settings,
        retrieval_tracer=retrieval_tracer,
        answering_tracer=answering_tracer,
    )

    assert services.chat_provider is chat_provider
    assert services.embedding_provider is embedding_provider
    assert services.vector_store is vector_store
    assert services.agent_service is not None
    assert services.document_import_service.max_upload_bytes == (settings.upload_max_bytes)

    document = _make_document()

    await services.indexing_service.index(
        ImportResult(
            status=ImportStatus.CREATED,
            document=document,
        )
    )

    retrieval_request = RetrievalQuery(
        query="vector search",
        top_k=1,
    )

    results = await services.retrieval_service.retrieve(retrieval_request)

    assert len(results) == 1
    assert results[0].chunk.document_id == document.document_id

    answer = await services.query_service.query(retrieval_request)

    assert answer.answer == "测试回答"
    assert answer.refused is True
    assert answer.citations == []
    assert answer.confidence == 0.0

    assert chat_provider.json_mode is True
    assert len(chat_provider.messages) == 2

    # 第一次直接调用 retrieval_service，
    # 第二次由 query_service 内部调用。
    assert len(retrieval_tracer.traces) == 2

    first_retrieval_trace = retrieval_tracer.traces[0]
    second_retrieval_trace = retrieval_tracer.traces[1]

    assert first_retrieval_trace.status is RetrievalTraceStatus.SUCCEEDED
    assert second_retrieval_trace.status is RetrievalTraceStatus.SUCCEEDED

    assert first_retrieval_trace.result_count == 1
    assert second_retrieval_trace.result_count == 1

    assert len(answering_tracer.traces) == 1

    answering_trace = answering_tracer.traces[0]

    assert answering_trace.status is AnsweringTraceStatus.SUCCEEDED
    assert answering_trace.model == "test-chat-model"
    assert answering_trace.refused is True
    assert answering_trace.citation_ids == []
    assert answering_trace.prompt_tokens == 10
    assert answering_trace.completion_tokens == 5
    assert answering_trace.total_tokens == 15
    assert answering_trace.error_type is None
