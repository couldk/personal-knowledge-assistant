import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.agent import (
    AgentRoute,
    KnowledgeAgentRequest,
    create_initial_agent_state,
)


def test_request_uses_safe_defaults() -> None:
    request = KnowledgeAgentRequest(
        question="什么是 RAG？",
    )

    assert request.question == "什么是 RAG？"
    assert request.top_k == 5
    assert request.filters == {}
    assert request.max_retrieval_attempts == 2
    assert request.min_relevance_score == 0.35


def test_request_normalizes_question_and_filters() -> None:
    request = KnowledgeAgentRequest(
        question="  什么是 RAG？  ",
        filters={
            "  file_name  ": "  rag.md  ",
        },
    )

    assert request.question == "什么是 RAG？"
    assert request.filters == {
        "file_name": "rag.md",
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("top_k", 0),
        ("top_k", 21),
        ("max_retrieval_attempts", 0),
        ("max_retrieval_attempts", 6),
        ("min_relevance_score", -1.1),
        ("min_relevance_score", 1.1),
    ],
)
def test_request_rejects_invalid_numeric_values(
    field_name: str,
    invalid_value: int | float,
) -> None:
    values: dict[str, object] = {
        "question": "什么是 RAG？",
        field_name: invalid_value,
    }

    with pytest.raises(ValidationError):
        KnowledgeAgentRequest.model_validate(values)


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
    ],
)
def test_request_rejects_blank_question(
    question: str,
) -> None:
    with pytest.raises(ValidationError):
        KnowledgeAgentRequest(
            question=question,
        )


@pytest.mark.parametrize(
    "filters",
    [
        {"": "rag.md"},
        {"   ": "rag.md"},
        {"file_name": ""},
        {"file_name": "   "},
    ],
)
def test_request_rejects_blank_filter_values(
    filters: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        KnowledgeAgentRequest(
            question="什么是 RAG？",
            filters=filters,
        )


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        KnowledgeAgentRequest.model_validate(
            {
                "question": "什么是 RAG？",
                "unknown_field": "value",
            }
        )


def test_create_initial_agent_state() -> None:
    request = KnowledgeAgentRequest(
        question="  什么是 RAG？  ",
        top_k=3,
        filters={
            "file_name": "rag.md",
        },
        max_retrieval_attempts=3,
        min_relevance_score=0.4,
    )

    state = create_initial_agent_state(request)

    assert state["question_history"] == ["什么是 RAG？"]
    assert state["original_question"] == "什么是 RAG？"
    assert state["retrieval_query"] == "什么是 RAG？"
    assert state["top_k"] == 3
    assert state["filters"] == {
        "file_name": "rag.md",
    }
    assert state["retrieval_attempts"] == 0
    assert state["max_retrieval_attempts"] == 3
    assert state["min_relevance_score"] == 0.4
    assert state["retrieval_results"] == []
    assert state["answer"] is None
    assert state["route"] is AgentRoute.RETRIEVE
    assert state["failure_reason"] is None
    assert state["error_type"] is None


def test_initial_state_copies_filters() -> None:
    request = KnowledgeAgentRequest(
        question="什么是 RAG？",
        filters={
            "file_name": "rag.md",
        },
    )

    state = create_initial_agent_state(request)
    request.filters["file_name"] = "changed.md"

    assert state["filters"] == {
        "file_name": "rag.md",
    }


def test_agent_route_values_are_stable() -> None:
    assert AgentRoute.RETRIEVE.value == "retrieve"
    assert AgentRoute.GRADE_EVIDENCE.value == "grade_evidence"
    assert AgentRoute.REWRITE_QUERY.value == "rewrite_query"
    assert AgentRoute.ANSWER.value == "answer"
    assert AgentRoute.REFUSE.value == "refuse"
    assert AgentRoute.HANDLE_ERROR.value == "handle_error"
    assert AgentRoute.END.value == "end"
