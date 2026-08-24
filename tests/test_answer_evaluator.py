from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest

from personal_knowledge_assistant.answering import (
    AnswerCitation,
    AnsweringTrace,
    AnsweringTraceStatus,
    EvidenceAnswer,
    InMemoryAnsweringTracer,
    InvalidAnswerResponseError,
    UnknownCitationError,
)
from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    RetrievalQuery,
    SearchResult,
)
from personal_knowledge_assistant.evaluation import (
    AnswerCaseResult,
    AnswerEvaluationCase,
    AnswerFailureReason,
    AnsweringEvaluator,
    EvaluationCategory,
)
from personal_knowledge_assistant.providers.exceptions import (
    ChatProviderError,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
    RetrievalHitTrace,
    RetrievalTrace,
    RetrievalTraceStatus,
)


class FakeRetrievalService:
    """能够生成检索结果和 Trace 的测试检索服务。"""

    def __init__(
        self,
        *,
        tracer: InMemoryRetrievalTracer,
        results_by_question: dict[
            str,
            Sequence[SearchResult],
        ],
        errors_by_question: dict[
            str,
            Exception,
        ]
        | None = None,
    ) -> None:
        self._tracer = tracer
        self._results_by_question = {
            question: list(results) for question, results in results_by_question.items()
        }
        self._errors_by_question = errors_by_question or {}
        self.requests: list[RetrievalQuery] = []

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        self.requests.append(request.model_copy(deep=True))

        query_hash = sha256(request.query.encode("utf-8")).hexdigest()

        error = self._errors_by_question.get(request.query)

        if error is not None:
            self._tracer.record(
                RetrievalTrace(
                    trace_id=uuid4(),
                    status=(RetrievalTraceStatus.FAILED),
                    query_hash=query_hash,
                    top_k=request.top_k,
                    result_count=0,
                    duration_ms=12.5,
                    embedding_model=("test-embedding-model"),
                    hits=[],
                    error_type=type(error).__name__,
                )
            )

            raise error

        results = [
            result.model_copy(deep=True)
            for result in self._results_by_question.get(
                request.query,
                [],
            )
        ]

        self._tracer.record(
            RetrievalTrace(
                trace_id=uuid4(),
                status=(RetrievalTraceStatus.SUCCEEDED),
                query_hash=query_hash,
                top_k=request.top_k,
                result_count=len(results),
                duration_ms=12.5,
                embedding_model=("test-embedding-model"),
                hits=[
                    RetrievalHitTrace(
                        chunk_id=(result.chunk.chunk_id),
                        score=result.score,
                    )
                    for result in results
                ],
                error_type=None,
            )
        )

        return results


class FakeAnsweringService:
    """能够生成回答和 Trace 的测试回答服务。"""

    def __init__(
        self,
        *,
        tracer: InMemoryAnsweringTracer,
        answers_by_question: dict[
            str,
            EvidenceAnswer,
        ],
        errors_by_question: dict[
            str,
            Exception,
        ]
        | None = None,
    ) -> None:
        self._tracer = tracer
        self._answers_by_question = answers_by_question
        self._errors_by_question = errors_by_question or {}
        self.questions: list[str] = []

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        del results

        self.questions.append(question)

        question_hash = sha256(question.encode("utf-8")).hexdigest()

        error = self._errors_by_question.get(question)

        if error is not None:
            self._tracer.record(
                AnsweringTrace(
                    trace_id=uuid4(),
                    status=(AnsweringTraceStatus.FAILED),
                    question_hash=question_hash,
                    duration_ms=30.0,
                    model="test-chat-model",
                    refused=None,
                    citation_ids=[],
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                    error_type=type(error).__name__,
                )
            )

            raise error

        answer = self._answers_by_question[question].model_copy(deep=True)

        self._tracer.record(
            AnsweringTrace(
                trace_id=uuid4(),
                status=(AnsweringTraceStatus.SUCCEEDED),
                question_hash=question_hash,
                duration_ms=30.0,
                model="test-chat-model",
                refused=answer.refused,
                citation_ids=[citation.chunk_id for citation in answer.citations],
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                error_type=None,
            )
        )

        return answer


def _make_result(
    *,
    file_name: str = "rag.md",
    chunk_character: str = "a",
    score: float = 0.95,
) -> SearchResult:
    """创建固定检索结果。"""

    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_character * 64,
            document_id=(f"file:///evaluation/{file_name}"),
            text=(f"Evaluation content from {file_name}."),
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path=(f"evaluation/documents/{file_name}"),
                file_name=file_name,
                document_type=(DocumentType.MARKDOWN),
                modified_at=datetime(
                    2026,
                    8,
                    13,
                    tzinfo=UTC,
                ),
                part_index=0,
                chunk_index=0,
            ),
        ),
        score=score,
    )


def _make_case(
    case_id: str = "q001",
    *,
    question: str = "RAG 如何工作？",
    expected_file_names: list[str] | None = None,
    expected_answer_terms: list[str] | None = None,
    expected_refused: bool = False,
    category: EvaluationCategory = (EvaluationCategory.DIRECT),
) -> AnswerEvaluationCase:
    """创建评测问题。"""

    return AnswerEvaluationCase(
        id=case_id,
        question=question,
        expected_file_names=(
            expected_file_names if expected_file_names is not None else ["rag.md"]
        ),
        expected_answer_terms=(
            expected_answer_terms if expected_answer_terms is not None else ["检索", "生成"]
        ),
        expected_refused=expected_refused,
        category=category,
        top_k=3,
    )


def _make_answer(
    *,
    answer: str = "RAG 先检索证据，然后生成回答。",
    citation_ids: Sequence[str] = ("a" * 64,),
    refused: bool = False,
    confidence: float = 0.9,
) -> EvidenceAnswer:
    """创建结构化回答。"""

    return EvidenceAnswer(
        answer=answer,
        citations=[
            AnswerCitation(
                chunk_id=chunk_id,
            )
            for chunk_id in citation_ids
        ],
        confidence=confidence,
        refused=refused,
    )


def _make_refusal() -> EvidenceAnswer:
    """创建合法拒答。"""

    return EvidenceAnswer(
        answer="现有证据不足，无法回答。",
        citations=[],
        confidence=0.0,
        refused=True,
    )


def _make_evaluator(
    *,
    results_by_question: dict[
        str,
        Sequence[SearchResult],
    ],
    answers_by_question: dict[
        str,
        EvidenceAnswer,
    ],
    retrieval_errors: dict[
        str,
        Exception,
    ]
    | None = None,
    answering_errors: dict[
        str,
        Exception,
    ]
    | None = None,
    callback_results: list[tuple[int, int, str]] | None = None,
    high_latency_threshold_ms: float = 5000.0,
) -> tuple[
    AnsweringEvaluator,
    FakeRetrievalService,
    FakeAnsweringService,
]:
    """创建带内存 Trace 的评测器。"""

    retrieval_tracer = InMemoryRetrievalTracer()
    answering_tracer = InMemoryAnsweringTracer()

    retrieval_service = FakeRetrievalService(
        tracer=retrieval_tracer,
        results_by_question=(results_by_question),
        errors_by_question=retrieval_errors,
    )
    answering_service = FakeAnsweringService(
        tracer=answering_tracer,
        answers_by_question=(answers_by_question),
        errors_by_question=answering_errors,
    )

    def on_case_completed(
        position: int,
        total: int,
        result: AnswerCaseResult,
    ) -> None:
        if callback_results is not None:
            callback_results.append(
                (
                    position,
                    total,
                    result.case_id,
                )
            )

    evaluator = AnsweringEvaluator(
        retrieval_service=retrieval_service,
        answering_service=answering_service,
        retrieval_tracer=retrieval_tracer,
        answering_tracer=answering_tracer,
        chat_model="test-chat-model",
        embedding_model=("test-embedding-model"),
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
        high_latency_threshold_ms=(high_latency_threshold_ms),
        on_case_completed=(on_case_completed if callback_results is not None else None),
    )

    return (
        evaluator,
        retrieval_service,
        answering_service,
    )


@pytest.mark.asyncio
async def test_evaluator_generates_perfect_report() -> None:
    case = _make_case()
    result = _make_result()

    evaluator, retrieval_service, answering_service = _make_evaluator(
        results_by_question={
            case.question: [result],
        },
        answers_by_question={
            case.question: _make_answer(),
        },
    )

    report = await evaluator.evaluate([case])

    assert report.case_count == 1
    assert report.chat_model == ("test-chat-model")
    assert report.embedding_model == ("test-embedding-model")
    assert report.success_rate == 1.0
    assert report.retrieval_recall_at_3 == 1.0
    assert report.citation_precision == 1.0
    assert report.citation_recall == 1.0
    assert report.answer_term_coverage == 1.0
    assert report.refusal_accuracy == 1.0

    case_result = report.cases[0]

    assert case_result.succeeded is True
    assert case_result.retrieved_file_names == ["rag.md"]
    assert case_result.cited_file_names == ["rag.md"]
    assert case_result.retrieval_duration_ms == 12.5
    assert case_result.answering_duration_ms == 30.0
    assert case_result.prompt_tokens == 100
    assert case_result.completion_tokens == 20
    assert case_result.total_tokens == 120
    assert case_result.failure_reasons == []
    assert case_result.error_type is None

    assert retrieval_service.requests[0].top_k == 3
    assert answering_service.questions == [case.question]


@pytest.mark.asyncio
async def test_evaluator_calculates_token_cost() -> None:
    case = _make_case()

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [_make_result()],
        },
        answers_by_question={
            case.question: _make_answer(),
        },
    )

    report = await evaluator.evaluate([case])

    # 100 / 1,000,000 * 2
    # + 20 / 1,000,000 * 8
    expected_cost = 0.00036

    assert report.cases[0].estimated_cost == (pytest.approx(expected_cost))
    assert report.estimated_total_cost == (pytest.approx(expected_cost))
    assert report.total_prompt_tokens == 100
    assert report.total_completion_tokens == 20
    assert report.total_tokens == 120


@pytest.mark.asyncio
async def test_evaluator_accepts_correct_refusal() -> None:
    case = _make_case(
        question="火星上有多少永久居民？",
        expected_file_names=[],
        expected_answer_terms=[],
        expected_refused=True,
        category=(EvaluationCategory.OUT_OF_SCOPE),
    )

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [],
        },
        answers_by_question={
            case.question: _make_refusal(),
        },
    )

    report = await evaluator.evaluate([case])
    result = report.cases[0]

    assert result.actual_refused is True
    assert result.refusal_correct is True
    assert result.retrieval_recall_at_3 == 1.0
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.answer_term_coverage == 1.0
    assert result.failure_reasons == []


@pytest.mark.asyncio
async def test_evaluator_classifies_retrieval_miss() -> None:
    case = _make_case()

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [
                _make_result(
                    file_name="security.md",
                    chunk_character="b",
                )
            ],
        },
        answers_by_question={
            case.question: _make_answer(
                answer="这是无关回答。",
                citation_ids=["b" * 64],
            ),
        },
    )

    report = await evaluator.evaluate([case])
    result = report.cases[0]

    assert result.retrieval_recall_at_3 == 0.0
    assert result.citation_precision == 0.0
    assert result.citation_recall == 0.0
    assert AnswerFailureReason.RETRIEVAL_MISS in result.failure_reasons
    assert AnswerFailureReason.IRRELEVANT_CITATION in result.failure_reasons
    assert AnswerFailureReason.MISSING_CITATION in result.failure_reasons


@pytest.mark.asyncio
async def test_evaluator_classifies_wrong_refusal() -> None:
    case = _make_case()

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [_make_result()],
        },
        answers_by_question={
            case.question: _make_refusal(),
        },
    )

    report = await evaluator.evaluate([case])
    result = report.cases[0]

    assert result.actual_refused is True
    assert result.refusal_correct is False
    assert AnswerFailureReason.WRONG_REFUSAL in result.failure_reasons
    assert AnswerFailureReason.MISSING_CITATION in result.failure_reasons


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_reason"),
    [
        (
            InvalidAnswerResponseError("Invalid JSON."),
            AnswerFailureReason.INVALID_MODEL_JSON,
        ),
        (
            UnknownCitationError(["b" * 64]),
            AnswerFailureReason.UNKNOWN_CITATION,
        ),
        (
            ChatProviderError("Provider unavailable."),
            AnswerFailureReason.PROVIDER_ERROR,
        ),
    ],
)
async def test_evaluator_classifies_answering_errors(
    error: Exception,
    expected_reason: AnswerFailureReason,
) -> None:
    case = _make_case()

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [_make_result()],
        },
        answers_by_question={},
        answering_errors={
            case.question: error,
        },
    )

    report = await evaluator.evaluate([case])
    result = report.cases[0]

    assert result.succeeded is False
    assert result.error_type == type(error).__name__
    assert expected_reason in result.failure_reasons
    assert result.prompt_tokens == 100
    assert result.total_tokens == 120


@pytest.mark.asyncio
async def test_retrieval_error_does_not_stop_next_case() -> None:
    failed_case = _make_case(
        case_id="q001",
        question="失败的问题",
    )
    successful_case = _make_case(
        case_id="q002",
        question="成功的问题",
    )

    evaluator, _, answering_service = _make_evaluator(
        results_by_question={
            successful_case.question: [_make_result()],
        },
        answers_by_question={
            successful_case.question: (_make_answer()),
        },
        retrieval_errors={
            failed_case.question: RuntimeError("Retrieval failed."),
        },
    )

    report = await evaluator.evaluate(
        [
            failed_case,
            successful_case,
        ]
    )

    assert report.case_count == 2
    assert report.cases[0].succeeded is False
    assert report.cases[0].error_type == ("RuntimeError")
    assert AnswerFailureReason.UNEXPECTED_ERROR in report.cases[0].failure_reasons

    assert report.cases[1].succeeded is True
    assert answering_service.questions == [successful_case.question]
    assert report.success_rate == 0.5


@pytest.mark.asyncio
async def test_callback_runs_for_every_case() -> None:
    first_case = _make_case(
        case_id="q001",
        question="问题一",
    )
    second_case = _make_case(
        case_id="q002",
        question="问题二",
    )

    callback_results: list[tuple[int, int, str]] = []

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            first_case.question: [_make_result()],
            second_case.question: [_make_result()],
        },
        answers_by_question={
            first_case.question: _make_answer(),
            second_case.question: _make_answer(),
        },
        callback_results=callback_results,
    )

    await evaluator.evaluate(
        [
            first_case,
            second_case,
        ]
    )

    assert callback_results == [
        (1, 2, "q001"),
        (2, 2, "q002"),
    ]


@pytest.mark.asyncio
async def test_evaluator_classifies_high_latency() -> None:
    case = _make_case()

    evaluator, _, _ = _make_evaluator(
        results_by_question={
            case.question: [_make_result()],
        },
        answers_by_question={
            case.question: _make_answer(),
        },
        high_latency_threshold_ms=0.000001,
    )

    report = await evaluator.evaluate([case])

    assert AnswerFailureReason.HIGH_LATENCY in report.cases[0].failure_reasons


@pytest.mark.asyncio
async def test_worst_cases_are_limited_and_ordered() -> None:
    cases = [
        _make_case(
            case_id=f"q{index:03d}",
            question=f"问题 {index}",
        )
        for index in range(1, 13)
    ]

    evaluator, _, _ = _make_evaluator(
        results_by_question={case.question: [_make_result()] for case in cases},
        answers_by_question={case.question: _make_answer() for case in cases},
    )

    report = await evaluator.evaluate(cases)

    assert len(report.cases) == 12
    assert len(report.worst_cases) == 10
    assert [result.case_id for result in report.worst_cases] == [
        f"q{index:03d}" for index in range(1, 11)
    ]


@pytest.mark.asyncio
async def test_evaluator_rejects_empty_cases() -> None:
    evaluator, _, _ = _make_evaluator(
        results_by_question={},
        answers_by_question={},
    )

    with pytest.raises(
        ValueError,
        match=("Answer evaluation cases cannot be empty"),
    ):
        await evaluator.evaluate([])


@pytest.mark.parametrize(
    (
        "input_price",
        "output_price",
        "latency_threshold",
        "message",
    ),
    [
        (
            -1.0,
            0.0,
            5000.0,
            "Input token price",
        ),
        (
            0.0,
            -1.0,
            5000.0,
            "Output token price",
        ),
        (
            0.0,
            0.0,
            0.0,
            "High latency threshold",
        ),
    ],
)
def test_evaluator_rejects_invalid_configuration(
    input_price: float,
    output_price: float,
    latency_threshold: float,
    message: str,
) -> None:
    retrieval_tracer = InMemoryRetrievalTracer()
    answering_tracer = InMemoryAnsweringTracer()

    retrieval_service = FakeRetrievalService(
        tracer=retrieval_tracer,
        results_by_question={},
    )
    answering_service = FakeAnsweringService(
        tracer=answering_tracer,
        answers_by_question={},
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        AnsweringEvaluator(
            retrieval_service=(retrieval_service),
            answering_service=(answering_service),
            retrieval_tracer=(retrieval_tracer),
            answering_tracer=(answering_tracer),
            chat_model="test-chat-model",
            embedding_model=("test-embedding-model"),
            input_cost_per_million=(input_price),
            output_cost_per_million=(output_price),
            high_latency_threshold_ms=(latency_threshold),
        )
