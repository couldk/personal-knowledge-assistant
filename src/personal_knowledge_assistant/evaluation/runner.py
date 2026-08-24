from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from math import fsum
from time import perf_counter
from typing import Protocol

from personal_knowledge_assistant.domain import RetrievalQuery, SearchResult
from personal_knowledge_assistant.evaluation.metrics import (
    calculate_retrieval_metrics,
)
from personal_knowledge_assistant.evaluation.models import (
    RetrievalCaseResult,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
)


class RetrievalServiceProtocol(Protocol):
    """评测执行器需要的最小检索接口。"""

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]: ...


CaseCompletedCallback = Callable[[int, int, RetrievalCaseResult], None]


class RetrievalEvaluator:
    """顺序执行检索问题并生成基线报告。"""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalServiceProtocol,
        model: str,
        dimension: int,
        chunk_size: int,
        chunk_overlap: int,
        on_case_completed: CaseCompletedCallback | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._model = model
        self._dimension = dimension
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._on_case_completed = on_case_completed

    async def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
    ) -> RetrievalEvaluationReport:
        """执行全部问题并汇总指标。"""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        case_results: list[RetrievalCaseResult] = []
        total = len(cases)

        for position, case in enumerate(cases, start=1):
            started_at = perf_counter()
            results = await self._retrieval_service.retrieve(
                RetrievalQuery(
                    query=case.question,
                    top_k=case.top_k,
                )
            )
            duration_ms = (perf_counter() - started_at) * 1000

            retrieved_file_names = [result.chunk.metadata.file_name for result in results]
            metrics = calculate_retrieval_metrics(
                expected_file_names=case.expected_file_names,
                retrieved_file_names=retrieved_file_names,
            )
            case_result = RetrievalCaseResult(
                case_id=case.id,
                category=case.category,
                expected_file_names=case.expected_file_names,
                retrieved_file_names=retrieved_file_names,
                retrieved_chunk_ids=[result.chunk.chunk_id for result in results],
                scores=[result.score for result in results],
                first_relevant_rank=metrics.first_relevant_rank,
                hit_at_1=metrics.hit_at_1,
                recall_at_3=metrics.recall_at_3,
                reciprocal_rank=metrics.reciprocal_rank,
                duration_ms=duration_ms,
            )
            case_results.append(case_result)

            if self._on_case_completed is not None:
                self._on_case_completed(position, total, case_result)

        count = len(case_results)

        return RetrievalEvaluationReport(
            generated_at=datetime.now(UTC),
            model=self._model,
            dimension=self._dimension,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            case_count=count,
            hit_at_1=fsum(result.hit_at_1 for result in case_results) / count,
            recall_at_3=fsum(result.recall_at_3 for result in case_results) / count,
            mean_reciprocal_rank=(fsum(result.reciprocal_rank for result in case_results) / count),
            average_duration_ms=(fsum(result.duration_ms for result in case_results) / count),
            cases=case_results,
        )
