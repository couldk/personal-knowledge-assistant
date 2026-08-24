import json
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from personal_knowledge_assistant.answering import (
    AnsweringService,
    AnsweringTraceStatus,
    EvidencePromptBuilder,
    InMemoryAnsweringTracer,
    InvalidAnswerResponseError,
    UnknownCitationError,
)
from personal_knowledge_assistant.domain.documents import (
    DocumentType,
)
from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    ChatResponse,
)
from personal_knowledge_assistant.domain.retrieval import (
    ChunkMetadata,
    DocumentChunk,
    SearchResult,
)
from personal_knowledge_assistant.providers.exceptions import (
    ChatProviderError,
)


class TraceChatProvider:
    def __init__(
        self,
        *,
        content: str,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.error = error

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        del messages
        del json_mode

        if self.error is not None:
            raise self.error

        return ChatResponse(
            content=self.content,
            model="test-chat-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


def _make_result() -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id="a" * 64,
            document_id="document-1",
            text="RAG combines retrieval and generation.",
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path="D:/private/knowledge.md",
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


def _answer_json(
    *,
    citation_id: str = "a" * 64,
) -> str:
    return json.dumps(
        {
            "answer": ("RAG combines retrieval and generation."),
            "citations": [
                {
                    "chunk_id": citation_id,
                }
            ],
            "confidence": 0.9,
            "refused": False,
        }
    )


def _make_service(
    *,
    content: str,
    tracer: InMemoryAnsweringTracer,
    error: Exception | None = None,
) -> AnsweringService:
    return AnsweringService(
        chat_provider=TraceChatProvider(
            content=content,
            error=error,
        ),
        prompt_builder=EvidencePromptBuilder(),
        tracer=tracer,
    )


@pytest.mark.asyncio
async def test_success_trace_contains_safe_metadata() -> None:
    question = "什么是 RAG？"
    tracer = InMemoryAnsweringTracer()
    service = _make_service(
        content=_answer_json(),
        tracer=tracer,
    )

    await service.answer(
        question=question,
        results=[_make_result()],
    )

    assert len(tracer.traces) == 1

    trace = tracer.traces[0]

    assert trace.status is AnsweringTraceStatus.SUCCEEDED
    assert trace.question_hash == sha256(question.encode("utf-8")).hexdigest()
    assert trace.model == "test-chat-model"
    assert trace.refused is False
    assert trace.citation_ids == ["a" * 64]
    assert trace.prompt_tokens == 10
    assert trace.completion_tokens == 5
    assert trace.total_tokens == 15
    assert trace.duration_ms >= 0
    assert trace.error_type is None

    serialized = trace.model_dump_json()

    assert question not in serialized
    assert "RAG combines retrieval and generation." not in serialized
    assert "D:/private" not in serialized


@pytest.mark.asyncio
async def test_invalid_json_records_failed_trace() -> None:
    tracer = InMemoryAnsweringTracer()
    service = _make_service(
        content="not valid JSON",
        tracer=tracer,
    )

    with pytest.raises(
        InvalidAnswerResponseError,
    ):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )

    trace = tracer.traces[0]

    assert trace.status is AnsweringTraceStatus.FAILED
    assert trace.model == "test-chat-model"
    assert trace.refused is None
    assert trace.citation_ids == []
    assert trace.error_type == ("InvalidAnswerResponseError")
    assert trace.total_tokens == 15


@pytest.mark.asyncio
async def test_unknown_citation_records_failed_trace() -> None:
    tracer = InMemoryAnsweringTracer()
    service = _make_service(
        content=_answer_json(
            citation_id="b" * 64,
        ),
        tracer=tracer,
    )

    with pytest.raises(UnknownCitationError):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )

    trace = tracer.traces[0]

    assert trace.status is AnsweringTraceStatus.FAILED
    assert trace.citation_ids == []
    assert trace.error_type == "UnknownCitationError"


@pytest.mark.asyncio
async def test_provider_error_records_failed_trace() -> None:
    tracer = InMemoryAnsweringTracer()
    service = _make_service(
        content="",
        tracer=tracer,
        error=ChatProviderError("Provider unavailable."),
    )

    with pytest.raises(ChatProviderError):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )

    trace = tracer.traces[0]

    assert trace.status is AnsweringTraceStatus.FAILED
    assert trace.model is None
    assert trace.prompt_tokens == 0
    assert trace.completion_tokens == 0
    assert trace.total_tokens == 0
    assert trace.error_type == "ChatProviderError"


def test_in_memory_tracer_returns_defensive_copies() -> None:
    tracer = InMemoryAnsweringTracer()

    first_result = tracer.traces

    first_result.clear()

    assert tracer.traces == []
