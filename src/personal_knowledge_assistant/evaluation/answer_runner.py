from collections.abc import (
    Callable,
    Sequence,
)
from datetime import UTC, datetime
from math import fsum
from time import perf_counter
from typing import Protocol

from personal_knowledge_assistant.answering import (
    AnsweringTrace,
    EvidenceAnswer,
    InMemoryAnsweringTracer,
    InvalidAnswerResponseError,
    UnknownCitationError,
)
from personal_knowledge_assistant.domain import (
    RetrievalQuery,
    SearchResult,
)
from personal_knowledge_assistant.evaluation.metrics import (
    calculate_answer_metrics,
)
from personal_knowledge_assistant.evaluation.models import (
    AnswerCaseResult,
    AnswerEvaluationCase,
    AnswerEvaluationReport,
    AnswerFailureReason,
)
from personal_knowledge_assistant.providers.exceptions import (
    ProviderError,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
    RetrievalTrace,
)


class AnswerRetrievalServiceProtocol(Protocol):
    """回答评测需要的检索服务接口。"""

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        """检索问题相关证据。"""
        ...


class AnsweringServiceProtocol(Protocol):
    """回答评测需要的回答服务接口。"""

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        """根据问题和检索证据生成回答。"""
        ...


AnswerCaseCompletedCallback = Callable[
    [int, int, AnswerCaseResult],
    None,
]


class AnsweringEvaluator:
    """顺序执行完整回答评测并生成报告。"""

    def __init__(
        self,
        *,
        retrieval_service: (AnswerRetrievalServiceProtocol),
        answering_service: AnsweringServiceProtocol,
        retrieval_tracer: InMemoryRetrievalTracer,
        answering_tracer: InMemoryAnsweringTracer,
        chat_model: str,
        embedding_model: str,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        high_latency_threshold_ms: float = 5000.0,
        on_case_completed: (AnswerCaseCompletedCallback | None) = None,
    ) -> None:
        if input_cost_per_million < 0:
            raise ValueError("Input token price cannot be negative.")

        if output_cost_per_million < 0:
            raise ValueError("Output token price cannot be negative.")

        if high_latency_threshold_ms <= 0:
            raise ValueError("High latency threshold must be positive.")

        self._retrieval_service = retrieval_service
        self._answering_service = answering_service
        self._retrieval_tracer = retrieval_tracer
        self._answering_tracer = answering_tracer
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._input_cost_per_million = input_cost_per_million
        self._output_cost_per_million = output_cost_per_million
        self._high_latency_threshold_ms = high_latency_threshold_ms
        self._on_case_completed = on_case_completed

    async def evaluate(
        self,
        cases: Sequence[AnswerEvaluationCase],
    ) -> AnswerEvaluationReport:
        """执行所有回答问题并汇总指标。"""

        if not cases:
            raise ValueError("Answer evaluation cases cannot be empty.")

        case_results: list[AnswerCaseResult] = []
        total = len(cases)

        for position, case in enumerate(
            cases,
            start=1,
        ):
            result = await self._evaluate_case(case)
            case_results.append(result)

            if self._on_case_completed is not None:
                self._on_case_completed(
                    position,
                    total,
                    result,
                )

        count = len(case_results)

        worst_cases = sorted(
            case_results,
            key=self._worst_case_sort_key,
        )[:10]

        return AnswerEvaluationReport(
            generated_at=datetime.now(UTC),
            chat_model=self._chat_model,
            embedding_model=self._embedding_model,
            case_count=count,
            success_rate=(sum(result.succeeded for result in case_results) / count),
            retrieval_recall_at_3=(
                fsum(result.retrieval_recall_at_3 for result in case_results) / count
            ),
            citation_precision=(fsum(result.citation_precision for result in case_results) / count),
            citation_recall=(fsum(result.citation_recall for result in case_results) / count),
            answer_term_coverage=(
                fsum(result.answer_term_coverage for result in case_results) / count
            ),
            refusal_accuracy=(sum(result.refusal_correct for result in case_results) / count),
            average_duration_ms=(fsum(result.duration_ms for result in case_results) / count),
            total_prompt_tokens=sum(result.prompt_tokens for result in case_results),
            total_completion_tokens=sum(result.completion_tokens for result in case_results),
            total_tokens=sum(result.total_tokens for result in case_results),
            estimated_total_cost=fsum(result.estimated_cost for result in case_results),
            cases=case_results,
            worst_cases=worst_cases,
        )

    async def _evaluate_case(
        self,
        case: AnswerEvaluationCase,
    ) -> AnswerCaseResult:
        """执行一个回答评测问题。"""

        started_at = perf_counter()
        retrieval_trace_count = len(self._retrieval_tracer.traces)
        answering_trace_count = len(self._answering_tracer.traces)

        results: list[SearchResult] = []
        answer: EvidenceAnswer | None = None
        error: Exception | None = None

        try:
            results = await self._retrieval_service.retrieve(
                RetrievalQuery(
                    query=case.question,
                    top_k=case.top_k,
                )
            )

            answer = await self._answering_service.answer(
                question=case.question,
                results=results,
            )
        except Exception as exc:
            error = exc

        duration_ms = (perf_counter() - started_at) * 1000

        retrieval_trace = self._new_retrieval_trace(retrieval_trace_count)
        answering_trace = self._new_answering_trace(answering_trace_count)

        retrieved_file_names = [result.chunk.metadata.file_name for result in results]
        retrieved_chunk_ids = [result.chunk.chunk_id for result in results]

        chunk_file_names = {
            result.chunk.chunk_id: result.chunk.metadata.file_name for result in results
        }

        if answer is not None:
            cited_chunk_ids = [citation.chunk_id for citation in answer.citations]
            cited_file_names = [
                chunk_file_names[chunk_id]
                for chunk_id in cited_chunk_ids
                if chunk_id in chunk_file_names
            ]

            metrics = calculate_answer_metrics(
                expected_file_names=(case.expected_file_names),
                expected_answer_terms=(case.expected_answer_terms),
                expected_refused=(case.expected_refused),
                retrieved_file_names=(retrieved_file_names),
                cited_file_names=cited_file_names,
                answer=answer.answer,
                actual_refused=answer.refused,
                recall_k=case.top_k,
            )

            retrieval_recall = metrics.retrieval_recall_at_3
            citation_precision = metrics.citation_precision
            citation_recall = metrics.citation_recall
            term_coverage = metrics.answer_term_coverage
            refusal_correct = metrics.refusal_correct
            actual_refused: bool | None = answer.refused
        else:
            cited_chunk_ids = []
            cited_file_names = []
            retrieval_recall = self._calculate_failed_retrieval_recall(
                expected_file_names=(case.expected_file_names),
                retrieved_file_names=(retrieved_file_names),
                top_k=case.top_k,
            )
            citation_precision = 0.0
            citation_recall = 0.0
            term_coverage = 0.0
            refusal_correct = False
            actual_refused = None

        prompt_tokens = answering_trace.prompt_tokens if answering_trace is not None else 0
        completion_tokens = answering_trace.completion_tokens if answering_trace is not None else 0
        total_tokens = answering_trace.total_tokens if answering_trace is not None else 0

        estimated_cost = self._calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        failure_reasons = self._classify_failures(
            case=case,
            answer=answer,
            error=error,
            retrieval_recall=retrieval_recall,
            citation_precision=(citation_precision),
            citation_recall=citation_recall,
            term_coverage=term_coverage,
            refusal_correct=refusal_correct,
            duration_ms=duration_ms,
        )

        return AnswerCaseResult(
            case_id=case.id,
            category=case.category,
            succeeded=error is None,
            expected_refused=(case.expected_refused),
            actual_refused=actual_refused,
            retrieved_file_names=(retrieved_file_names),
            retrieved_chunk_ids=(retrieved_chunk_ids),
            cited_file_names=cited_file_names,
            cited_chunk_ids=cited_chunk_ids,
            retrieval_recall_at_3=(retrieval_recall),
            citation_precision=(citation_precision),
            citation_recall=citation_recall,
            answer_term_coverage=term_coverage,
            refusal_correct=refusal_correct,
            duration_ms=duration_ms,
            retrieval_duration_ms=(
                retrieval_trace.duration_ms if retrieval_trace is not None else 0.0
            ),
            answering_duration_ms=(
                answering_trace.duration_ms if answering_trace is not None else 0.0
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            failure_reasons=failure_reasons,
            error_type=(type(error).__name__ if error is not None else None),
        )

    def _new_retrieval_trace(
        self,
        previous_count: int,
    ) -> RetrievalTrace | None:
        """返回当前问题新增的检索 Trace。"""

        traces = self._retrieval_tracer.traces

        if len(traces) <= previous_count:
            return None

        return traces[-1]

    def _new_answering_trace(
        self,
        previous_count: int,
    ) -> AnsweringTrace | None:
        """返回当前问题新增的回答 Trace。"""

        traces = self._answering_tracer.traces

        if len(traces) <= previous_count:
            return None

        return traces[-1]

    def _calculate_cost(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """根据每百万 Token 单价估算成本。"""

        return (
            prompt_tokens / 1_000_000 * self._input_cost_per_million
            + completion_tokens / 1_000_000 * self._output_cost_per_million
        )

    @staticmethod
    def _calculate_failed_retrieval_recall(
        *,
        expected_file_names: Sequence[str],
        retrieved_file_names: Sequence[str],
        top_k: int,
    ) -> float:
        """回答失败时仍计算检索召回率。"""

        if not expected_file_names:
            return 1.0

        expected = set(expected_file_names)
        retrieved = list(dict.fromkeys(retrieved_file_names))[:top_k]

        return len(expected & set(retrieved)) / len(expected)

    def _classify_failures(
        self,
        *,
        case: AnswerEvaluationCase,
        answer: EvidenceAnswer | None,
        error: Exception | None,
        retrieval_recall: float,
        citation_precision: float,
        citation_recall: float,
        term_coverage: float,
        refusal_correct: bool,
        duration_ms: float,
    ) -> list[AnswerFailureReason]:
        """对单个问题的失败原因分类。"""

        reasons: list[AnswerFailureReason] = []

        if retrieval_recall < 1.0:
            reasons.append(AnswerFailureReason.RETRIEVAL_MISS)

        if not refusal_correct:
            reasons.append(AnswerFailureReason.WRONG_REFUSAL)

        if not case.expected_refused and (
            answer is None or not answer.citations or citation_recall < 1.0
        ):
            reasons.append(AnswerFailureReason.MISSING_CITATION)

        if citation_precision < 1.0:
            reasons.append(AnswerFailureReason.IRRELEVANT_CITATION)

        if term_coverage < 1.0:
            reasons.append(AnswerFailureReason.LOW_ANSWER_TERM_COVERAGE)

        if isinstance(
            error,
            InvalidAnswerResponseError,
        ):
            reasons.append(AnswerFailureReason.INVALID_MODEL_JSON)
        elif isinstance(
            error,
            UnknownCitationError,
        ):
            reasons.append(AnswerFailureReason.UNKNOWN_CITATION)
        elif isinstance(error, ProviderError):
            reasons.append(AnswerFailureReason.PROVIDER_ERROR)
        elif error is not None:
            reasons.append(AnswerFailureReason.UNEXPECTED_ERROR)

        if duration_ms > self._high_latency_threshold_ms:
            reasons.append(AnswerFailureReason.HIGH_LATENCY)

        return list(dict.fromkeys(reasons))

    @staticmethod
    def _worst_case_sort_key(
        result: AnswerCaseResult,
    ) -> tuple[float, str]:
        """为最差问题生成确定性排序键。"""

        quality_score = (
            result.retrieval_recall_at_3
            + result.citation_precision
            + result.citation_recall
            + result.answer_term_coverage
            + float(result.refusal_correct)
        ) / 5

        if not result.succeeded:
            quality_score = 0.0

        return (
            quality_score,
            result.case_id,
        )
