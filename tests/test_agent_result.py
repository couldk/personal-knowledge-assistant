from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.agent import (
    AgentEvidence,
    AgentOutcome,
    AgentState,
    KnowledgeAgentResult,
    create_agent_result,
)
from personal_knowledge_assistant.answering import (
    AnswerCitation,
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    SearchResult,
)


def _search_result(
    *,
    chunk_character: str = "a",
    score: float = 0.95,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_character * 64,
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
                page_number=2,
                section_path=[
                    "Agent",
                    "RAG",
                ],
            ),
        ),
        score=score,
    )


def _answered_answer(
    *,
    chunk_character: str = "a",
) -> EvidenceAnswer:
    return EvidenceAnswer(
        answer=("RAG combines retrieval with generation."),
        citations=[
            AnswerCitation(
                chunk_id=chunk_character * 64,
            )
        ],
        confidence=0.9,
        refused=False,
    )


def _refused_answer() -> EvidenceAnswer:
    return EvidenceAnswer(
        answer=("当前知识库中没有足够证据回答该问题。"),
        citations=[],
        confidence=0.0,
        refused=True,
    )


def test_create_answered_agent_result() -> None:
    state = AgentState(
        original_question="什么是RAG？",
        retrieval_query="RAG 检索增强生成",
        retrieval_attempts=2,
        retrieval_results=[
            _search_result(),
        ],
        answer=_answered_answer(),
        failure_reason=None,
        error_type=None,
    )

    result = create_agent_result(state)

    assert result.outcome is AgentOutcome.ANSWERED
    assert result.answer == _answered_answer()
    assert result.retrieval_attempts == 2
    assert result.final_retrieval_query == ("RAG 检索增强生成")
    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.chunk_id == "a" * 64
    assert evidence.file_name == "rag.md"
    assert evidence.page_number == 2
    assert evidence.section_path == [
        "Agent",
        "RAG",
    ]
    assert evidence.score == 0.95

    # 对外证据摘要不能包含完整Chunk正文。
    assert "text" not in evidence.model_dump()


def test_create_refused_agent_result() -> None:
    state = AgentState(
        retrieval_query="不存在的资料",
        retrieval_attempts=2,
        retrieval_results=[],
        answer=_refused_answer(),
        failure_reason=("Retrieved evidence was insufficient."),
        error_type=None,
    )

    result = create_agent_result(state)

    assert result.outcome is AgentOutcome.REFUSED
    assert result.answer.refused is True
    assert result.evidence == []
    assert result.error_type is None


def test_create_failed_agent_result() -> None:
    state = AgentState(
        retrieval_query="什么是RAG？",
        retrieval_attempts=1,
        retrieval_results=[],
        answer=_refused_answer(),
        failure_reason=("Knowledge retrieval failed."),
        error_type="TimeoutError",
    )

    result = create_agent_result(state)

    assert result.outcome is AgentOutcome.FAILED
    assert result.answer.refused is True
    assert result.failure_reason == ("Knowledge retrieval failed.")
    assert result.error_type == "TimeoutError"


def test_create_result_requires_answer() -> None:
    state = AgentState(
        retrieval_query="什么是RAG？",
        retrieval_attempts=1,
    )

    with pytest.raises(
        ValueError,
        match="final answer",
    ):
        create_agent_result(state)


def test_create_result_requires_retrieval_query() -> None:
    state = AgentState(
        retrieval_attempts=1,
        answer=_refused_answer(),
    )

    with pytest.raises(
        ValueError,
        match="retrieval query",
    ):
        create_agent_result(state)


def test_create_result_requires_retrieval_attempts() -> None:
    state = AgentState(
        retrieval_query="什么是RAG？",
        answer=_refused_answer(),
    )

    with pytest.raises(
        ValueError,
        match="retrieval_attempts",
    ):
        create_agent_result(state)


def test_result_rejects_unknown_citation() -> None:
    with pytest.raises(
        ValidationError,
        match="must reference returned",
    ):
        KnowledgeAgentResult(
            outcome=AgentOutcome.ANSWERED,
            answer=_answered_answer(
                chunk_character="b",
            ),
            retrieval_attempts=1,
            final_retrieval_query="什么是RAG？",
            evidence=[
                AgentEvidence(
                    chunk_id="a" * 64,
                    file_name="rag.md",
                    page_number=2,
                    section_path=["RAG"],
                    score=0.95,
                )
            ],
            failure_reason=None,
            error_type=None,
        )


def test_result_rejects_duplicate_evidence() -> None:
    evidence = AgentEvidence(
        chunk_id="a" * 64,
        file_name="rag.md",
        page_number=2,
        section_path=["RAG"],
        score=0.95,
    )

    with pytest.raises(
        ValidationError,
        match="must be unique",
    ):
        KnowledgeAgentResult(
            outcome=AgentOutcome.ANSWERED,
            answer=_answered_answer(),
            retrieval_attempts=1,
            final_retrieval_query="什么是RAG？",
            evidence=[
                evidence,
                evidence.model_copy(),
            ],
            failure_reason=None,
            error_type=None,
        )


def test_answered_outcome_rejects_refused_answer() -> None:
    with pytest.raises(
        ValidationError,
        match="cannot contain a refused answer",
    ):
        KnowledgeAgentResult(
            outcome=AgentOutcome.ANSWERED,
            answer=_refused_answer(),
            retrieval_attempts=1,
            final_retrieval_query="什么是RAG？",
            evidence=[],
            failure_reason=None,
            error_type=None,
        )


def test_failed_outcome_requires_error_information() -> None:
    with pytest.raises(
        ValidationError,
        match="requires an error type",
    ):
        KnowledgeAgentResult(
            outcome=AgentOutcome.FAILED,
            answer=_refused_answer(),
            retrieval_attempts=1,
            final_retrieval_query="什么是RAG？",
            evidence=[],
            failure_reason=None,
            error_type=None,
        )
