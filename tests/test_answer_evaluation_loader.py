import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.evaluation import (
    AnswerEvaluationCase,
    EvaluationCategory,
    EvaluationDataError,
    load_answer_evaluation_cases,
    validate_answer_expected_files,
)


def _case_payload(
    case_id: str,
    *,
    question: str = "What is RAG?",
    expected_file_names: list[str] | None = None,
    expected_answer_terms: list[str] | None = None,
    expected_refused: bool = False,
    category: str = "direct",
) -> dict[str, object]:
    return {
        "id": case_id,
        "question": question,
        "expected_file_names": (
            expected_file_names if expected_file_names is not None else ["rag.md"]
        ),
        "expected_answer_terms": (
            expected_answer_terms
            if expected_answer_terms is not None
            else [
                "retrieval",
                "generation",
            ]
        ),
        "expected_refused": expected_refused,
        "category": category,
        "top_k": 3,
    }


def _write_jsonl(
    path: Path,
    payloads: Sequence[dict[str, object]],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_accepts_answerable_case() -> None:
    case = AnswerEvaluationCase.model_validate(_case_payload("q001"))

    assert case.id == "q001"
    assert case.category is EvaluationCategory.DIRECT
    assert case.expected_refused is False
    assert case.expected_file_names == ["rag.md"]


def test_model_normalizes_strings() -> None:
    case = AnswerEvaluationCase.model_validate(
        _case_payload(
            "q001",
            question="  What is RAG?  ",
            expected_file_names=[" rag.md "],
            expected_answer_terms=[
                " retrieval ",
                " generation ",
            ],
        )
    )

    assert case.question == "What is RAG?"
    assert case.expected_file_names == ["rag.md"]
    assert case.expected_answer_terms == [
        "retrieval",
        "generation",
    ]


def test_model_accepts_refusal_case() -> None:
    case = AnswerEvaluationCase.model_validate(
        _case_payload(
            "q001",
            expected_file_names=[],
            expected_answer_terms=[],
            expected_refused=True,
            category="out_of_scope",
        )
    )

    assert case.expected_refused is True
    assert case.category is EvaluationCategory.OUT_OF_SCOPE


@pytest.mark.parametrize(
    "payload",
    [
        _case_payload(
            "q001",
            expected_file_names=[],
        ),
        _case_payload(
            "q001",
            expected_answer_terms=[],
        ),
        _case_payload(
            "q001",
            expected_refused=True,
            category="direct",
        ),
        _case_payload(
            "q001",
            expected_file_names=[],
            expected_answer_terms=[],
            category="out_of_scope",
        ),
        _case_payload(
            "q001",
            expected_file_names=["rag.md"],
            expected_answer_terms=[],
            expected_refused=True,
            category="out_of_scope",
        ),
    ],
)
def test_model_rejects_inconsistent_state(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationCase.model_validate(payload)


def test_model_rejects_duplicate_files() -> None:
    with pytest.raises(
        ValidationError,
        match=("Expected file names must be unique"),
    ):
        AnswerEvaluationCase.model_validate(
            _case_payload(
                "q001",
                expected_file_names=[
                    "rag.md",
                    "rag.md",
                ],
            )
        )


def test_model_rejects_duplicate_terms() -> None:
    with pytest.raises(
        ValidationError,
        match=("Expected answer terms must be unique"),
    ):
        AnswerEvaluationCase.model_validate(
            _case_payload(
                "q001",
                expected_answer_terms=[
                    "RAG",
                    "RAG",
                ],
            )
        )


def test_model_rejects_blank_question() -> None:
    with pytest.raises(
        ValidationError,
        match="Question cannot be blank",
    ):
        AnswerEvaluationCase.model_validate(
            _case_payload(
                "q001",
                question="   ",
            )
        )


def test_model_rejects_unknown_fields() -> None:
    payload = _case_payload("q001")
    payload["unexpected_field"] = "value"

    with pytest.raises(ValidationError):
        AnswerEvaluationCase.model_validate(payload)


def test_cross_document_case_requires_two_files() -> None:
    with pytest.raises(
        ValidationError,
        match="at least two expected files",
    ):
        AnswerEvaluationCase.model_validate(
            _case_payload(
                "q001",
                expected_file_names=["rag.md"],
                category="cross_document",
            )
        )


def test_loader_reads_valid_cases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answering.jsonl"

    _write_jsonl(
        path,
        [
            _case_payload("q001"),
            _case_payload("q002"),
        ],
    )

    cases = load_answer_evaluation_cases(
        path,
        expected_count=2,
    )

    assert [case.id for case in cases] == [
        "q001",
        "q002",
    ]


def test_loader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answering.jsonl"
    path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDataError,
        match=("Invalid answer evaluation case at line 1"),
    ):
        load_answer_evaluation_cases(
            path,
            expected_count=1,
        )


def test_loader_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answering.jsonl"

    _write_jsonl(
        path,
        [
            _case_payload("q001"),
            _case_payload("q001"),
        ],
    )

    with pytest.raises(
        EvaluationDataError,
        match=("Duplicate answer evaluation case ID"),
    ):
        load_answer_evaluation_cases(
            path,
            expected_count=2,
        )


def test_loader_requires_exact_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answering.jsonl"

    _write_jsonl(
        path,
        [
            _case_payload("q001"),
        ],
    )

    with pytest.raises(
        EvaluationDataError,
        match=("Expected 2 answer evaluation cases, received 1"),
    ):
        load_answer_evaluation_cases(
            path,
            expected_count=2,
        )


def test_loader_requires_sequential_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answering.jsonl"

    _write_jsonl(
        path,
        [
            _case_payload("q002"),
            _case_payload("q001"),
        ],
    )

    with pytest.raises(
        EvaluationDataError,
        match="sequential and ordered",
    ):
        load_answer_evaluation_cases(
            path,
            expected_count=2,
        )


def test_validate_answer_expected_files() -> None:
    cases = [
        AnswerEvaluationCase.model_validate(
            _case_payload(
                "q001",
                expected_file_names=["missing.md"],
            )
        )
    ]

    with pytest.raises(
        EvaluationDataError,
        match="missing.md",
    ):
        validate_answer_expected_files(
            cases,
            {"rag.md"},
        )


def test_committed_answer_dataset_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]
    questions_path = project_root / "evaluation" / "answering_questions.jsonl"
    documents_directory = project_root / "evaluation" / "documents"

    cases = load_answer_evaluation_cases(
        questions_path,
        expected_count=30,
    )

    validate_answer_expected_files(
        cases,
        {path.name for path in documents_directory.glob("*.md")},
    )

    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    assert sum(case.expected_refused for case in cases) == 5
