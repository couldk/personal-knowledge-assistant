from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.answering import (
    AnswerCitation,
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain.documents import (
    DocumentType,
)
from personal_knowledge_assistant.domain.retrieval import (
    ChunkMetadata,
    DocumentChunk,
    RetrievalQuery,
    SearchResult,
)
from personal_knowledge_assistant.querying import (
    KnowledgeQueryService,
    create_query_service,
)


class FakeRetriever:
    """用于查询服务测试的检索器。"""

    def __init__(
        self,
        *,
        results: list[SearchResult],
        error: Exception | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return list(self.results)


class FakeAnswerGenerator:
    """用于查询服务测试的回答生成器。"""

    def __init__(
        self,
        *,
        answer: EvidenceAnswer,
        error: Exception | None = None,
    ) -> None:
        self.generated_answer = answer
        self.error = error
        self.questions: list[str] = []
        self.received_results: list[list[SearchResult]] = []

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        self.questions.append(question)
        self.received_results.append(list(results))

        if self.error is not None:
            raise self.error

        return self.generated_answer


def _make_result() -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id="a" * 64,
            document_id="document-1",
            text="RAG combines retrieval and generation.",
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path="D:/documents/knowledge.md",
                file_name="knowledge.md",
                document_type=DocumentType.MARKDOWN,
                modified_at=datetime(
                    2026,
                    1,
                    1,
                    tzinfo=UTC,
                ),
                part_index=0,
                chunk_index=0,
                page_number=1,
                section_path=["Agent", "RAG"],
            ),
        ),
        score=0.95,
    )


def _make_answer() -> EvidenceAnswer:
    return EvidenceAnswer(
        answer="RAG combines retrieval and generation.",
        citations=[
            AnswerCitation(
                chunk_id="a" * 64,
            )
        ],
        confidence=0.9,
        refused=False,
    )


def _make_refusal() -> EvidenceAnswer:
    return EvidenceAnswer(
        answer="现有证据不足，无法回答。",
        citations=[],
        confidence=0.0,
        refused=True,
    )


@pytest.mark.asyncio
async def test_query_retrieves_and_generates_answer() -> None:
    result = _make_result()
    retriever = FakeRetriever(results=[result])
    answer_generator = FakeAnswerGenerator(answer=_make_answer())

    service = KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    request = RetrievalQuery(
        query="什么是 RAG？",
        top_k=5,
    )

    answer = await service.query(request)

    assert answer == _make_answer()
    assert retriever.requests == [request]
    assert answer_generator.questions == ["什么是 RAG？"]
    assert answer_generator.received_results == [[result]]


@pytest.mark.asyncio
async def test_query_passes_filters_to_retriever() -> None:
    retriever = FakeRetriever(results=[_make_result()])
    answer_generator = FakeAnswerGenerator(answer=_make_answer())

    service = KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    request = RetrievalQuery(
        query="什么是 RAG？",
        top_k=3,
        filters={
            "document_type": "markdown",
        },
    )

    await service.query(request)

    assert retriever.requests[0].top_k == 3
    assert retriever.requests[0].filters == {"document_type": "markdown"}


@pytest.mark.asyncio
async def test_query_allows_refusal_for_empty_results() -> None:
    retriever = FakeRetriever(results=[])
    answer_generator = FakeAnswerGenerator(answer=_make_refusal())

    service = KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    answer = await service.query(
        RetrievalQuery(
            query="没有资料的问题。",
        )
    )

    assert answer.refused is True
    assert answer.citations == []
    assert answer_generator.received_results == [[]]


@pytest.mark.asyncio
async def test_query_propagates_retrieval_error() -> None:
    retriever = FakeRetriever(
        results=[],
        error=RuntimeError("Retrieval failed."),
    )
    answer_generator = FakeAnswerGenerator(answer=_make_refusal())

    service = KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval failed",
    ):
        await service.query(
            RetrievalQuery(
                query="什么是 RAG？",
            )
        )

    assert answer_generator.questions == []


@pytest.mark.asyncio
async def test_query_propagates_answering_error() -> None:
    retriever = FakeRetriever(results=[_make_result()])
    answer_generator = FakeAnswerGenerator(
        answer=_make_answer(),
        error=RuntimeError("Answering failed."),
    )

    service = KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Answering failed",
    ):
        await service.query(
            RetrievalQuery(
                query="什么是 RAG？",
            )
        )


def test_factory_creates_query_service() -> None:
    retriever = FakeRetriever(results=[_make_result()])
    answer_generator = FakeAnswerGenerator(answer=_make_answer())

    service = create_query_service(
        retriever=retriever,
        answer_generator=answer_generator,
    )

    assert isinstance(service, KnowledgeQueryService)
