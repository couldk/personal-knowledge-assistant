from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.agent import (
    AgentOutcome,
    KnowledgeAgentRequest,
    KnowledgeAgentService,
    create_knowledge_agent_service,
)
from personal_knowledge_assistant.answering import (
    AnswerCitation,
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain import (
    ChatMessage,
    ChatResponse,
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    RetrievalQuery,
    SearchResult,
)


class StaticRetriever:
    """始终返回固定证据的测试检索器。"""

    def __init__(
        self,
        result: SearchResult,
    ) -> None:
        self._result = result
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request)

        return [
            self._result.model_copy(deep=True),
        ]


class StaticAnswerGenerator:
    """始终返回固定回答的测试生成器。"""

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        del question, results

        return EvidenceAnswer(
            answer=("RAG combines retrieval with generation."),
            citations=[
                AnswerCitation(
                    chunk_id="a" * 64,
                )
            ],
            confidence=0.9,
            refused=False,
        )


class UnusedChatProvider:
    """证据充分时不应被调用的 Chat Provider。"""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        del messages, json_mode

        raise AssertionError("Chat provider should not be called.")


def _search_result() -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id="a" * 64,
            document_id="document-1",
            text=("RAG combines retrieval with language model generation."),
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path="D:/documents/rag.md",
                file_name="rag.md",
                document_type=DocumentType.MARKDOWN,
                modified_at=datetime(
                    2026,
                    8,
                    17,
                    tzinfo=UTC,
                ),
                part_index=0,
                chunk_index=0,
                page_number=1,
                section_path=["RAG"],
            ),
        ),
        score=0.95,
    )


def _service() -> KnowledgeAgentService:
    return create_knowledge_agent_service(
        retriever=StaticRetriever(
            _search_result(),
        ),
        answer_generator=StaticAnswerGenerator(),
        chat_provider=UnusedChatProvider(),
    )


def _request(
    question: str,
) -> KnowledgeAgentRequest:
    return KnowledgeAgentRequest(
        question=question,
        top_k=3,
        max_retrieval_attempts=2,
        min_relevance_score=0.35,
    )


@pytest.mark.asyncio
async def test_service_returns_public_agent_result() -> None:
    service = _service()

    result = await service.run(
        _request("什么是RAG？"),
        thread_id="session-1",
    )

    assert result.outcome is AgentOutcome.ANSWERED
    assert result.answer.refused is False
    assert result.retrieval_attempts == 1
    assert result.final_retrieval_query == "什么是RAG？"
    assert len(result.evidence) == 1


@pytest.mark.asyncio
async def test_same_thread_accumulates_question_history() -> None:
    service = _service()

    await service.run(
        _request("什么是RAG？"),
        thread_id="shared-session",
    )
    await service.run(
        _request("RAG如何减少幻觉？"),
        thread_id="shared-session",
    )

    history = await service.get_question_history(
        thread_id="shared-session",
    )

    assert history == [
        "什么是RAG？",
        "RAG如何减少幻觉？",
    ]


@pytest.mark.asyncio
async def test_different_threads_are_isolated() -> None:
    service = _service()

    await service.run(
        _request("线程A的问题"),
        thread_id="session-a",
    )
    await service.run(
        _request("线程B的问题"),
        thread_id="session-b",
    )

    session_a_history = await service.get_question_history(
        thread_id="session-a",
    )
    session_b_history = await service.get_question_history(
        thread_id="session-b",
    )

    assert session_a_history == [
        "线程A的问题",
    ]
    assert session_b_history == [
        "线程B的问题",
    ]


@pytest.mark.asyncio
async def test_unknown_thread_has_empty_history() -> None:
    service = _service()

    history = await service.get_question_history(
        thread_id="unknown-session",
    )

    assert history == []


@pytest.mark.asyncio
async def test_service_rejects_empty_thread_id() -> None:
    service = _service()

    with pytest.raises(
        ValueError,
        match="thread_id cannot be empty",
    ):
        await service.run(
            _request("什么是RAG？"),
            thread_id="   ",
        )
