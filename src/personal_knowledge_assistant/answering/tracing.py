import logging
from typing import Protocol, runtime_checkable

from personal_knowledge_assistant.answering.trace_models import (
    AnsweringTrace,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class AnsweringTracer(Protocol):
    """回答 Trace 统一记录接口。"""

    def record(
        self,
        trace: AnsweringTrace,
    ) -> None:
        """记录一次回答 Trace。"""
        ...


class LoggingAnsweringTracer:
    """将回答 Trace 写入应用日志。"""

    def record(
        self,
        trace: AnsweringTrace,
    ) -> None:
        logger.info(
            "answering_trace",
            extra={"answering_trace": trace.model_dump(mode="json")},
        )


class InMemoryAnsweringTracer:
    """在内存中保存 Trace，主要用于测试。"""

    def __init__(self) -> None:
        self._traces: list[AnsweringTrace] = []

    @property
    def traces(self) -> list[AnsweringTrace]:
        """返回 Trace 的防御性副本。"""

        return [trace.model_copy(deep=True) for trace in self._traces]

    def record(
        self,
        trace: AnsweringTrace,
    ) -> None:
        """保存 Trace 的深拷贝。"""

        self._traces.append(trace.model_copy(deep=True))
