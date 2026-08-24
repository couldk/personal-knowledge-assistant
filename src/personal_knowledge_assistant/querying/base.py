from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from personal_knowledge_assistant.answering.models import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain.retrieval import (
    RetrievalQuery,
    SearchResult,
)


@runtime_checkable
class Retriever(Protocol):
    """检索服务统一接口。"""

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        """根据请求检索证据。"""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """基于证据回答问题的统一接口。"""

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        """根据问题和证据生成回答。"""
        ...
