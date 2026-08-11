from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest

from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    RetrievalQuery,
    SearchResult,
    create_chunk_id,
)
from personal_knowledge_assistant.providers.base import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
    LoggingRetrievalTracer,
    RetrievalHitTrace,
    RetrievalService,
    RetrievalTrace,
    RetrievalTracer,
    RetrievalTraceStatus,
)


class FakeEmbeddingProvider:
    def __init__(
        self,
        query_embedding: Sequence[float],
        *,
        error: Exception | None = None,
    ) -> None:
        self._query_embedding = list(query_embedding)
        self._error = error
        self.queries: list[str] = []

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._query_embedding.copy() for _ in texts]

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        self.queries.append(query)

        if self._error is not None:
            raise self._error

        return self._query_embedding.copy()


class FakeVectorStore:
    def __init__(
        self,
        results: Sequence[SearchResult],
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results)
        self._error = error
        self.search_calls: list[
            tuple[
                list[float],
                int,
                dict[str, str] | None,
            ]
        ] = []

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        del chunks, embeddings

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        self.search_calls.append(
            (
                list(query_embedding),
                limit,
                filters.copy() if filters is not None else None,
            )
        )

        if self._error is not None:
            raise self._error

        return [result.model_copy(deep=True) for result in self._results]

    async def deactivate_document(
        self,
        document_id: str,
    ) -> int:
        del document_id
        return 0


def _make_chunk(
    *,
    document_id: str,
    content_hash: str,
    chunk_index: int,
    file_name: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=create_chunk_id(
            document_id=document_id,
            content_hash=content_hash,
            part_index=0,
            chunk_index=chunk_index,
        ),
        document_id=document_id,
        text=text,
        metadata=ChunkMetadata(
            content_hash=content_hash,
            source_path=f"D:/documents/{file_name}",
            file_name=file_name,
            document_type=DocumentType.TEXT,
            modified_at=datetime(
                2026,
                8,
                11,
                10,
                0,
                tzinfo=UTC,
            ),
            part_index=0,
            chunk_index=chunk_index,
        ),
    )


def _make_results() -> list[SearchResult]:
    first_chunk = _make_chunk(
        document_id="document-a",
        content_hash="a" * 64,
        chunk_index=0,
        file_name="vector.txt",
        text="Vector retrieval.",
    )
    second_chunk = _make_chunk(
        document_id="document-b",
        content_hash="b" * 64,
        chunk_index=0,
        file_name="memory.txt",
        text="Long-term memory.",
    )

    return [
        SearchResult(
            chunk=first_chunk,
            score=0.95,
        ),
        SearchResult(
            chunk=second_chunk,
            score=0.72,
        ),
    ]


def _make_service(
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    tracer: RetrievalTracer,
) -> RetrievalService:
    return RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tracer=tracer,
        embedding_model=("Qwen/Qwen3-Embedding-0.6B"),
    )


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_forwards_search_options() -> None:
    embedding_provider = FakeEmbeddingProvider([1.0, 0.0, 0.0])
    vector_store = FakeVectorStore(_make_results())
    tracer = InMemoryRetrievalTracer()
    service = _make_service(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tracer=tracer,
    )
    request = RetrievalQuery(
        query="如何进行向量检索？",
        top_k=2,
        filters={
            "file_name": "vector.txt",
        },
    )

    results = await service.retrieve(request)

    assert embedding_provider.queries == ["如何进行向量检索？"]
    assert vector_store.search_calls == [
        (
            [1.0, 0.0, 0.0],
            2,
            {"file_name": "vector.txt"},
        )
    ]
    assert results == _make_results()


@pytest.mark.asyncio
async def test_success_trace_contains_hits_without_raw_query() -> None:
    query = "这是不能写进 Trace 的私人问题"
    expected_results = _make_results()
    tracer = InMemoryRetrievalTracer()
    service = _make_service(
        embedding_provider=FakeEmbeddingProvider([1.0, 0.0, 0.0]),
        vector_store=FakeVectorStore(expected_results),
        tracer=tracer,
    )

    await service.retrieve(
        RetrievalQuery(
            query=query,
            top_k=2,
        )
    )

    assert len(tracer.traces) == 1
    trace = tracer.traces[0]
    assert trace.status == (RetrievalTraceStatus.SUCCEEDED)
    assert trace.query_hash == sha256(query.encode("utf-8")).hexdigest()
    assert trace.top_k == 2
    assert trace.result_count == 2
    assert trace.duration_ms >= 0
    assert trace.embedding_model == ("Qwen/Qwen3-Embedding-0.6B")
    assert [hit.chunk_id for hit in trace.hits] == [
        result.chunk.chunk_id for result in expected_results
    ]
    assert [hit.score for hit in trace.hits] == [0.95, 0.72]
    assert query not in trace.model_dump_json()


@pytest.mark.asyncio
async def test_empty_results_produce_success_trace() -> None:
    tracer = InMemoryRetrievalTracer()
    service = _make_service(
        embedding_provider=FakeEmbeddingProvider([1.0, 0.0, 0.0]),
        vector_store=FakeVectorStore([]),
        tracer=tracer,
    )

    results = await service.retrieve(
        RetrievalQuery(
            query="No matching document",
            top_k=5,
        )
    )

    assert results == []
    trace = tracer.traces[0]
    assert trace.status == (RetrievalTraceStatus.SUCCEEDED)
    assert trace.result_count == 0
    assert trace.hits == []
    assert trace.error_type is None


@pytest.mark.asyncio
async def test_embedding_failure_records_trace_without_search() -> None:
    embedding_provider = FakeEmbeddingProvider(
        [],
        error=RuntimeError("embedding unavailable"),
    )
    vector_store = FakeVectorStore([])
    tracer = InMemoryRetrievalTracer()
    service = _make_service(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tracer=tracer,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding unavailable",
    ):
        await service.retrieve(
            RetrievalQuery(
                query="Vector search",
                top_k=3,
            )
        )

    assert vector_store.search_calls == []
    trace = tracer.traces[0]
    assert trace.status == RetrievalTraceStatus.FAILED
    assert trace.result_count == 0
    assert trace.hits == []
    assert trace.error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_vector_store_failure_records_failed_trace() -> None:
    vector_store = FakeVectorStore(
        [],
        error=RuntimeError("store unavailable"),
    )
    tracer = InMemoryRetrievalTracer()
    service = _make_service(
        embedding_provider=FakeEmbeddingProvider([1.0, 0.0, 0.0]),
        vector_store=vector_store,
        tracer=tracer,
    )

    with pytest.raises(
        RuntimeError,
        match="store unavailable",
    ):
        await service.retrieve(
            RetrievalQuery(
                query="Vector search",
                top_k=3,
            )
        )

    assert len(vector_store.search_calls) == 1
    trace = tracer.traces[0]
    assert trace.status == RetrievalTraceStatus.FAILED
    assert trace.error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_logging_tracer_does_not_log_raw_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    query = "Private retrieval question"
    service = _make_service(
        embedding_provider=FakeEmbeddingProvider([1.0, 0.0, 0.0]),
        vector_store=FakeVectorStore([]),
        tracer=LoggingRetrievalTracer(),
    )

    with caplog.at_level(
        "INFO",
        logger=("personal_knowledge_assistant.retrieval.tracing"),
    ):
        await service.retrieve(
            RetrievalQuery(
                query=query,
                top_k=5,
            )
        )

    assert "retrieval_completed" in caplog.text
    assert query not in caplog.text


def test_in_memory_tracer_returns_defensive_copies() -> None:
    tracer = InMemoryRetrievalTracer()
    original_trace = RetrievalTrace(
        trace_id=uuid4(),
        status=RetrievalTraceStatus.SUCCEEDED,
        query_hash="a" * 64,
        top_k=1,
        result_count=1,
        duration_ms=1.0,
        embedding_model="test-model",
        hits=[
            RetrievalHitTrace(
                chunk_id="chunk-a",
                score=0.9,
            )
        ],
    )

    tracer.record(original_trace)
    returned_traces = tracer.traces
    returned_traces[0].hits.clear()

    assert len(tracer.traces) == 1
    assert len(tracer.traces[0].hits) == 1
    assert tracer.traces[0].hits[0].chunk_id == ("chunk-a")


def test_fake_components_implement_provider_protocols() -> None:
    embedding_provider = FakeEmbeddingProvider([1.0, 0.0, 0.0])
    vector_store = FakeVectorStore([])

    assert isinstance(
        embedding_provider,
        EmbeddingProvider,
    )
    assert isinstance(
        vector_store,
        VectorStoreProvider,
    )
