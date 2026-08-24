from personal_knowledge_assistant.answering.models import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain.retrieval import (
    RetrievalQuery,
)
from personal_knowledge_assistant.querying.base import (
    AnswerGenerator,
    Retriever,
)


class KnowledgeQueryService:
    """编排检索和基于证据回答的应用服务。"""

    def __init__(
        self,
        *,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator

    async def query(
        self,
        request: RetrievalQuery,
    ) -> EvidenceAnswer:
        """检索相关证据并生成结构化回答。"""

        results = await self._retriever.retrieve(request)

        return await self._answer_generator.answer(
            question=request.query,
            results=results,
        )
