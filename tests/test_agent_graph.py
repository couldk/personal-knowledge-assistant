from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast

import pytest

from personal_knowledge_assistant.agent import (
    AgentOutcome,
    AgentState,
    KnowledgeAgentGraph,
    KnowledgeAgentNodes,
    KnowledgeAgentRequest,
    create_agent_result,
    create_initial_agent_state,
    create_knowledge_agent_graph,
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


class SequenceRetriever:
    """按照预设顺序返回多批检索结果。"""

    def __init__(
        self,
        result_batches: Sequence[Sequence[SearchResult]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._result_batches = [list(batch) for batch in result_batches]
        self._error = error
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request)

        if self._error is not None:
            raise self._error

        if not self._result_batches:
            return []

        batch = self._result_batches.pop(0)

        return [result.model_copy(deep=True) for result in batch]


class StaticAnswerGenerator:
    """返回固定结构化答案。"""

    def __init__(
        self,
        answer: EvidenceAnswer,
    ) -> None:
        self._answer = answer
        self.questions: list[str] = []
        self.result_batches: list[list[SearchResult]] = []

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        self.questions.append(question)
        self.result_batches.append(list(results))

        return self._answer.model_copy(deep=True)


class StaticChatProvider:
    """返回固定的查询改写结果。"""

    def __init__(
        self,
        rewritten_query: str,
    ) -> None:
        self._rewritten_query = rewritten_query
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        self.calls += 1

        return ChatResponse(
            content=self._rewritten_query,
            model="test-chat-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


def _search_result(
    *,
    score: float,
) -> SearchResult:
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
        score=score,
    )


def _answer() -> EvidenceAnswer:
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


def _request(
    *,
    max_retrieval_attempts: int = 2,
) -> KnowledgeAgentRequest:
    return KnowledgeAgentRequest(
        question="什么是RAG？",
        top_k=3,
        filters={
            "file_name": "rag.md",
        },
        max_retrieval_attempts=(max_retrieval_attempts),
        min_relevance_score=0.35,
    )


def _graph(
    *,
    result_batches: Sequence[Sequence[SearchResult]],
    retrieval_error: Exception | None = None,
) -> tuple[
    KnowledgeAgentGraph,
    SequenceRetriever,
    StaticAnswerGenerator,
    StaticChatProvider,
]:
    retriever = SequenceRetriever(
        result_batches,
        error=retrieval_error,
    )
    answer_generator = StaticAnswerGenerator(
        _answer(),
    )
    chat_provider = StaticChatProvider(
        "RAG 检索增强生成 工作原理",
    )

    nodes = KnowledgeAgentNodes(
        retriever=retriever,
        answer_generator=answer_generator,
        chat_provider=chat_provider,
    )

    graph = create_knowledge_agent_graph(
        nodes=nodes,
    )

    return (
        graph,
        retriever,
        answer_generator,
        chat_provider,
    )


@pytest.mark.asyncio
async def test_graph_answers_when_first_retrieval_is_sufficient() -> None:
    graph, retriever, answer_generator, chat_provider = _graph(
        result_batches=[
            [_search_result(score=0.95)],
        ],
    )

    final_state = cast(
        AgentState,
        await graph.ainvoke(
            create_initial_agent_state(
                _request(),
            )
        ),
    )
    result = create_agent_result(final_state)

    assert result.outcome is AgentOutcome.ANSWERED
    assert result.retrieval_attempts == 1
    assert result.final_retrieval_query == "什么是RAG？"
    assert len(result.evidence) == 1

    assert len(retriever.requests) == 1
    assert len(answer_generator.questions) == 1
    assert chat_provider.calls == 0


@pytest.mark.asyncio
async def test_graph_rewrites_query_then_answers() -> None:
    graph, retriever, answer_generator, chat_provider = _graph(
        result_batches=[
            [_search_result(score=0.10)],
            [_search_result(score=0.90)],
        ],
    )

    final_state = cast(
        AgentState,
        await graph.ainvoke(
            create_initial_agent_state(
                _request(),
            )
        ),
    )
    result = create_agent_result(final_state)

    assert result.outcome is AgentOutcome.ANSWERED
    assert result.retrieval_attempts == 2
    assert result.final_retrieval_query == ("RAG 检索增强生成 工作原理")

    assert len(retriever.requests) == 2
    assert retriever.requests[1].query == ("RAG 检索增强生成 工作原理")
    assert answer_generator.questions == [
        "什么是RAG？",
    ]
    assert chat_provider.calls == 1


@pytest.mark.asyncio
async def test_graph_refuses_after_attempts_are_exhausted() -> None:
    graph, retriever, answer_generator, chat_provider = _graph(
        result_batches=[
            [_search_result(score=0.10)],
            [_search_result(score=0.20)],
        ],
    )

    final_state = cast(
        AgentState,
        await graph.ainvoke(
            create_initial_agent_state(
                _request(),
            )
        ),
    )
    result = create_agent_result(final_state)

    assert result.outcome is AgentOutcome.REFUSED
    assert result.answer.refused is True
    assert result.answer.citations == []
    assert result.retrieval_attempts == 2
    assert result.failure_reason is not None

    assert len(retriever.requests) == 2
    assert answer_generator.questions == []
    assert chat_provider.calls == 1


@pytest.mark.asyncio
async def test_graph_converts_retrieval_error_to_safe_failure() -> None:
    graph, retriever, answer_generator, chat_provider = _graph(
        result_batches=[],
        retrieval_error=RuntimeError(
            "secret internal error",
        ),
    )

    final_state = cast(
        AgentState,
        await graph.ainvoke(
            create_initial_agent_state(
                _request(),
            )
        ),
    )
    result = create_agent_result(final_state)

    assert result.outcome is AgentOutcome.FAILED
    assert result.answer.refused is True
    assert result.error_type == "RuntimeError"
    assert result.failure_reason == ("Knowledge retrieval failed.")

    assert "secret internal error" not in (result.answer.answer)
    assert len(retriever.requests) == 1
    assert answer_generator.questions == []
    assert chat_provider.calls == 0
