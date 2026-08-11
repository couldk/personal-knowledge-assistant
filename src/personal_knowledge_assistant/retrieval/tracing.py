import logging
from typing import Protocol, runtime_checkable

from personal_knowledge_assistant.retrieval.models import (
    RetrievalTrace,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class RetrievalTracer(Protocol):
    """检索 Trace 统一记录接口。"""

    def record(
        self,
        trace: RetrievalTrace,
    ) -> None:
        """记录一次检索 Trace。"""
        ...


class LoggingRetrievalTracer:
    """把检索 Trace 写入应用日志。"""

    def record(
        self,
        trace: RetrievalTrace,
    ) -> None:
        """以结构化字段记录检索信息。"""

        logger.info(
            "retrieval_completed",
            extra={
                "trace_id": str(trace.trace_id),
                "retrieval_status": (trace.status.value),
                "query_hash": trace.query_hash,
                "top_k": trace.top_k,
                "result_count": (trace.result_count),
                "duration_ms": trace.duration_ms,
                "embedding_model": (trace.embedding_model),
                "chunk_ids": [hit.chunk_id for hit in trace.hits],
                "scores": [hit.score for hit in trace.hits],
                "error_type": trace.error_type,
            },
        )


class InMemoryRetrievalTracer:
    """在内存中保存 Trace，主要用于测试。"""

    def __init__(self) -> None:
        self._traces: list[RetrievalTrace] = []

    @property
    def traces(self) -> list[RetrievalTrace]:
        """返回 Trace 的防御性副本。"""

        return [trace.model_copy(deep=True) for trace in self._traces]

    def record(
        self,
        trace: RetrievalTrace,
    ) -> None:
        """保存 Trace 的深拷贝。"""

        self._traces.append(trace.model_copy(deep=True))
