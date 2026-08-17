from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.agent import (
    AgentRoute,
    AgentState,
    KnowledgeAgentNodes,
    KnowledgeAgentRequest,
    create_initial_agent_state,
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
    MessageRole,
    RetrievalQuery,
    SearchResult,
)


class FakeRetriever:
    """测试用检索服务。"""

    def __init__(
        self,
        results: Sequence[SearchResult],
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results)
        self._error = error
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request)

        if self._error is not None:
            raise self._error

        return [result.model_copy(deep=True) for result in self._results]


class FakeAnswerGenerator:
    """测试用回答服务。"""

    def __init__(
        self,
        answer: EvidenceAnswer,
        *,
        error: Exception | None = None,
    ) -> None:
        self._answer = answer
        self._error = error
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

        if self._error is not None:
            raise self._error

        return self._answer.model_copy(deep=True)


class FakeChatProvider:
    """测试用问题改写Provider。"""

    def __init__(
        self,
        content: str,
        *,
        error: Exception | None = None,
    ) -> None:
        self._content = content
        self._error = error
        self.messages: list[ChatMessage] = []
        self.json_mode: bool | None = None

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        self.messages = list(messages)
        self.json_mode = json_mode

        if self._error is not None:
            raise self._error

        return ChatResponse(
            content=self._content,
            model="test-chat-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


def _search_result(
    *,
    score: float = 0.95,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id="a" * 64,
            document_id="document-1",
            text=("RAG combines retrieval with generation."),
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path=("D:/documents/rag.md"),
                file_name="rag.md",
                document_type=(DocumentType.MARKDOWN),
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


def _answered_answer() -> EvidenceAnswer:
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


def _refused_answer() -> EvidenceAnswer:
    return EvidenceAnswer(
        answer=("当前证据不足，无法回答。"),
        citations=[],
        confidence=0.0,
        refused=True,
    )


def _initial_state() -> AgentState:
    return create_initial_agent_state(
        KnowledgeAgentRequest(
            question="什么是RAG？",
            top_k=3,
            filters={
                "file_name": "rag.md",
            },
            max_retrieval_attempts=2,
            min_relevance_score=0.35,
        )
    )


def _nodes(
    *,
    results: Sequence[SearchResult] = (),
    retrieval_error: Exception | None = None,
    answer: EvidenceAnswer | None = None,
    answer_error: Exception | None = None,
    rewritten_query: str = ("RAG 检索增强生成 工作原理"),
    rewrite_error: Exception | None = None,
) -> tuple[
    KnowledgeAgentNodes,
    FakeRetriever,
    FakeAnswerGenerator,
    FakeChatProvider,
]:
    retriever = FakeRetriever(
        results,
        error=retrieval_error,
    )
    answer_generator = FakeAnswerGenerator(
        answer or _answered_answer(),
        error=answer_error,
    )
    chat_provider = FakeChatProvider(
        rewritten_query,
        error=rewrite_error,
    )

    nodes = KnowledgeAgentNodes(
        retriever=retriever,
        answer_generator=answer_generator,
        chat_provider=chat_provider,
    )

    return (
        nodes,
        retriever,
        answer_generator,
        chat_provider,
    )


@pytest.mark.asyncio
async def test_retrieve_forwards_request_and_increments_attempts() -> None:
    expected_result = _search_result()

    nodes, retriever, _, _ = _nodes(
        results=[
            expected_result,
        ]
    )

    update = await nodes.retrieve(
        _initial_state(),
    )

    assert len(retriever.requests) == 1

    request = retriever.requests[0]

    assert request.query == "什么是RAG？"
    assert request.top_k == 3
    assert request.filters == {
        "file_name": "rag.md",
    }

    assert update["retrieval_attempts"] == 1
    assert update["retrieval_results"] == [
        expected_result,
    ]
    assert update["route"] is AgentRoute.GRADE_EVIDENCE
    assert update["failure_reason"] is None
    assert update["error_type"] is None


@pytest.mark.asyncio
async def test_retrieve_failure_returns_safe_error_state() -> None:
    nodes, _, _, _ = _nodes(retrieval_error=RuntimeError("private retrieval details"))

    update = await nodes.retrieve(
        _initial_state(),
    )

    assert update["retrieval_attempts"] == 1
    assert update["route"] is AgentRoute.HANDLE_ERROR
    assert update["failure_reason"] == ("Knowledge retrieval failed.")
    assert update["error_type"] == "RuntimeError"

    assert "private retrieval details" not in str(update)


def test_grade_evidence_routes_sufficient_results_to_answer() -> None:
    nodes, _, _, _ = _nodes()

    state = _initial_state()
    state["retrieval_attempts"] = 1
    state["retrieval_results"] = [
        _search_result(score=0.9),
    ]

    update = nodes.grade_evidence(state)

    assert update["route"] is AgentRoute.ANSWER
    assert update["failure_reason"] is None


def test_grade_evidence_routes_low_score_to_rewrite() -> None:
    nodes, _, _, _ = _nodes()

    state = _initial_state()
    state["retrieval_attempts"] = 1
    state["retrieval_results"] = [
        _search_result(score=0.2),
    ]

    update = nodes.grade_evidence(state)

    assert update["route"] is AgentRoute.REWRITE_QUERY
    assert update["failure_reason"] == ("Retrieved evidence was insufficient.")


def test_grade_evidence_routes_exhausted_attempts_to_refusal() -> None:
    nodes, _, _, _ = _nodes()

    state = _initial_state()
    state["retrieval_attempts"] = 2
    state["retrieval_results"] = []

    update = nodes.grade_evidence(state)

    assert update["route"] is AgentRoute.REFUSE
    assert "maximum number" in (update["failure_reason"] or "")


@pytest.mark.asyncio
async def test_rewrite_query_uses_chat_provider() -> None:
    nodes, _, _, chat_provider = _nodes(rewritten_query=('  "RAG 语义检索 工作流程"  '))

    update = await nodes.rewrite_query(
        _initial_state(),
    )

    assert update["retrieval_query"] == ("RAG 语义检索 工作流程")
    assert update["retrieval_results"] == []
    assert update["route"] is AgentRoute.RETRIEVE

    assert chat_provider.json_mode is False
    assert len(chat_provider.messages) == 2
    assert chat_provider.messages[0].role is MessageRole.SYSTEM
    assert chat_provider.messages[1].role is MessageRole.USER
    assert "什么是RAG？" in chat_provider.messages[1].content


@pytest.mark.asyncio
async def test_empty_rewritten_query_routes_to_error() -> None:
    nodes, _, _, _ = _nodes(
        rewritten_query="   ",
    )

    update = await nodes.rewrite_query(
        _initial_state(),
    )

    assert update["route"] is AgentRoute.HANDLE_ERROR
    assert update["error_type"] == ("EmptyRewrittenQueryError")


@pytest.mark.asyncio
async def test_answer_uses_original_question_and_results() -> None:
    search_result = _search_result()

    nodes, _, answer_generator, _ = _nodes(
        results=[
            search_result,
        ]
    )

    state = _initial_state()
    state["retrieval_query"] = "已经改写的问题"
    state["retrieval_results"] = [
        search_result,
    ]

    update = await nodes.answer(state)

    assert answer_generator.questions == [
        "什么是RAG？",
    ]
    assert answer_generator.result_batches == [
        [
            search_result,
        ]
    ]

    assert update["answer"] == (_answered_answer())
    assert update["route"] is AgentRoute.END
    assert update["failure_reason"] is None
    assert update["error_type"] is None


@pytest.mark.asyncio
async def test_answer_failure_routes_to_error() -> None:
    nodes, _, _, _ = _nodes(answer_error=RuntimeError("private answering details"))

    update = await nodes.answer(
        _initial_state(),
    )

    assert update["route"] is AgentRoute.HANDLE_ERROR
    assert update["failure_reason"] == ("Evidence-grounded answering failed.")
    assert update["error_type"] == "RuntimeError"

    assert "private answering details" not in str(update)


def test_refuse_returns_safe_structured_answer() -> None:
    nodes, _, _, _ = _nodes()

    state = _initial_state()
    state["failure_reason"] = "Retrieved evidence was insufficient."

    update = nodes.refuse(state)

    answer = update["answer"]

    assert answer is not None
    assert answer.refused is True
    assert answer.citations == []
    assert answer.confidence == 0.0
    assert update["route"] is AgentRoute.END
    assert update["error_type"] is None


def test_handle_error_returns_safe_structured_answer() -> None:
    nodes, _, _, _ = _nodes()

    state = _initial_state()
    state["failure_reason"] = "Knowledge retrieval failed."
    state["error_type"] = "TimeoutError"

    update = nodes.handle_error(state)

    answer = update["answer"]

    assert answer is not None
    assert answer.refused is True
    assert answer.citations == []
    assert update["route"] is AgentRoute.END
    assert update["failure_reason"] == ("Knowledge retrieval failed.")
    assert update["error_type"] == "TimeoutError"
