import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    RetrievalQuery,
    SearchResult,
    create_chunk_id,
)
from personal_knowledge_assistant.evaluation import (
    EvaluationCategory,
    EvaluationDataError,
    RetrievalEvaluationCase,
    RetrievalEvaluator,
    calculate_retrieval_metrics,
    load_evaluation_cases,
    unique_in_order,
    validate_expected_files,
)


class FakeRetrievalService:
    def __init__(
        self,
        results_by_query: dict[str, Sequence[SearchResult]],
    ) -> None:
        self._results_by_query = {
            query: list(results) for query, results in results_by_query.items()
        }
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request.model_copy(deep=True))

        return [result.model_copy(deep=True) for result in self._results_by_query[request.query]]


def _case_payload(
    case_id: str,
    *,
    question: str = "What is RAG?",
    expected_file_names: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": case_id,
        "question": question,
        "expected_file_names": (expected_file_names or ["rag.md"]),
        "category": "direct",
        "top_k": 3,
    }


def _write_jsonl(
    path: Path,
    payloads: Sequence[dict[str, object]],
) -> None:
    path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads) + "\n",
        encoding="utf-8",
    )


def _make_result(
    *,
    file_name: str,
    score: float,
    chunk_index: int = 0,
) -> SearchResult:
    document_id = f"file:///evaluation/{file_name}"
    content_hash = {
        "rag.md": "a" * 64,
        "vector-search.md": "b" * 64,
        "memory.md": "c" * 64,
    }.get(file_name, "d" * 64)
    chunk = DocumentChunk(
        chunk_id=create_chunk_id(
            document_id=document_id,
            content_hash=content_hash,
            part_index=0,
            chunk_index=chunk_index,
        ),
        document_id=document_id,
        text=f"Evaluation content from {file_name}.",
        metadata=ChunkMetadata(
            content_hash=content_hash,
            source_path=f"evaluation/documents/{file_name}",
            file_name=file_name,
            document_type=DocumentType.MARKDOWN,
            modified_at=datetime(2026, 8, 11, 10, 0, tzinfo=UTC),
            part_index=0,
            chunk_index=chunk_index,
        ),
    )

    return SearchResult(chunk=chunk, score=score)


def test_loader_reads_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _write_jsonl(
        path,
        [
            _case_payload("q001"),
            _case_payload("q002"),
        ],
    )

    cases = load_evaluation_cases(path, expected_count=2)

    assert [case.id for case in cases] == ["q001", "q002"]


def test_loader_requires_exact_case_count(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _write_jsonl(path, [_case_payload("q001")])

    with pytest.raises(
        EvaluationDataError,
        match="Expected 2 evaluation cases, received 1",
    ):
        load_evaluation_cases(path, expected_count=2)


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _write_jsonl(
        path,
        [
            _case_payload("q001"),
            _case_payload("q001"),
        ],
    )

    with pytest.raises(
        EvaluationDataError,
        match="Duplicate evaluation case ID: q001",
    ):
        load_evaluation_cases(path, expected_count=2)


def test_loader_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(
        EvaluationDataError,
        match="Invalid evaluation case at line 1",
    ):
        load_evaluation_cases(path, expected_count=1)


def test_expected_files_must_exist() -> None:
    cases = [
        RetrievalEvaluationCase.model_validate(
            _case_payload(
                "q001",
                expected_file_names=["missing.md"],
            )
        )
    ]

    with pytest.raises(
        EvaluationDataError,
        match="Expected documents are missing: missing.md",
    ):
        validate_expected_files(cases, {"rag.md"})


def test_committed_baseline_has_30_valid_cases() -> None:
    project_root = Path(__file__).resolve().parents[1]
    questions_path = project_root / "evaluation" / "retrieval_questions.jsonl"
    documents_directory = project_root / "evaluation" / "documents"

    cases = load_evaluation_cases(
        questions_path,
        expected_count=30,
    )
    validate_expected_files(
        cases,
        {path.name for path in documents_directory.glob("*.md")},
    )

    assert len({case.id for case in cases}) == 30
    assert len(list(documents_directory.glob("*.md"))) == 5


def test_unique_in_order_removes_duplicate_document_hits() -> None:
    assert unique_in_order(["rag.md", "rag.md", "memory.md", "rag.md"]) == ["rag.md", "memory.md"]


@pytest.mark.parametrize(
    (
        "retrieved",
        "expected_rank",
        "expected_hit",
        "expected_recall",
        "expected_mrr",
    ),
    [
        (["rag.md"], 1, 1.0, 1.0, 1.0),
        (["vector-search.md", "rag.md"], 2, 0.0, 1.0, 0.5),
        (["vector-search.md", "memory.md"], None, 0.0, 0.0, 0.0),
    ],
)
def test_single_document_metrics(
    retrieved: list[str],
    expected_rank: int | None,
    expected_hit: float,
    expected_recall: float,
    expected_mrr: float,
) -> None:
    metrics = calculate_retrieval_metrics(
        expected_file_names=["rag.md"],
        retrieved_file_names=retrieved,
    )

    assert metrics.first_relevant_rank == expected_rank
    assert metrics.hit_at_1 == expected_hit
    assert metrics.recall_at_3 == expected_recall
    assert metrics.reciprocal_rank == expected_mrr


def test_multi_document_recall_uses_unique_top_three_files() -> None:
    metrics = calculate_retrieval_metrics(
        expected_file_names=["rag.md", "vector-search.md"],
        retrieved_file_names=[
            "rag.md",
            "rag.md",
            "memory.md",
            "vector-search.md",
        ],
    )

    assert metrics.hit_at_1 == 1.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.reciprocal_rank == 1.0


@pytest.mark.asyncio
async def test_evaluator_returns_case_results_and_aggregates() -> None:
    first_question = "What is RAG?"
    second_question = "Why must dimensions match?"
    retrieval_service = FakeRetrievalService(
        {
            first_question: [
                _make_result(
                    file_name="vector-search.md",
                    score=0.9,
                ),
                _make_result(
                    file_name="rag.md",
                    score=0.8,
                ),
            ],
            second_question: [
                _make_result(
                    file_name="vector-search.md",
                    score=0.95,
                )
            ],
        }
    )
    completed: list[str] = []
    evaluator = RetrievalEvaluator(
        retrieval_service=retrieval_service,
        model="test-model",
        dimension=3,
        chunk_size=64,
        chunk_overlap=8,
        on_case_completed=(
            lambda position, total, result: completed.append(f"{position}/{total}:{result.case_id}")
        ),
    )
    cases = [
        RetrievalEvaluationCase(
            id="q001",
            question=first_question,
            expected_file_names=["rag.md"],
            category=EvaluationCategory.DIRECT,
            top_k=3,
        ),
        RetrievalEvaluationCase(
            id="q002",
            question=second_question,
            expected_file_names=["vector-search.md"],
            category=EvaluationCategory.MULTILINGUAL,
            top_k=3,
        ),
    ]

    report = await evaluator.evaluate(cases)

    assert report.case_count == 2
    assert report.hit_at_1 == pytest.approx(0.5)
    assert report.recall_at_3 == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx(0.75)
    assert report.average_duration_ms >= 0
    assert report.model == "test-model"
    assert report.dimension == 3
    assert completed == ["1/2:q001", "2/2:q002"]
    assert report.cases[0].retrieved_file_names == [
        "vector-search.md",
        "rag.md",
    ]
    assert len(report.cases[0].retrieved_chunk_ids) == 2
    assert report.cases[0].scores == [0.9, 0.8]
    assert [request.top_k for request in retrieval_service.requests] == [3, 3]
