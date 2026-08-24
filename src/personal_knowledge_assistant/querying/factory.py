from personal_knowledge_assistant.querying.base import (
    AnswerGenerator,
    Retriever,
)
from personal_knowledge_assistant.querying.service import (
    KnowledgeQueryService,
)


def create_query_service(
    *,
    retriever: Retriever,
    answer_generator: AnswerGenerator,
) -> KnowledgeQueryService:
    """使用已有服务创建完整查询流水线。"""

    return KnowledgeQueryService(
        retriever=retriever,
        answer_generator=answer_generator,
    )
