from enum import StrEnum
from operator import add
from typing import Annotated, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from personal_knowledge_assistant.answering import EvidenceAnswer
from personal_knowledge_assistant.domain import SearchResult

NonEmptyText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]


class AgentRoute(StrEnum):
    """知识 Agent 支持的路由状态。"""

    RETRIEVE = "retrieve"
    GRADE_EVIDENCE = "grade_evidence"
    REWRITE_QUERY = "rewrite_query"
    ANSWER = "answer"
    REFUSE = "refuse"
    HANDLE_ERROR = "handle_error"
    END = "end"


class KnowledgeAgentRequest(BaseModel):
    """执行一次知识 Agent 查询所需的输入。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    question: NonEmptyText
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    filters: dict[NonEmptyText, NonEmptyText] = Field(
        default_factory=dict,
    )
    max_retrieval_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
    )
    min_relevance_score: float = Field(
        default=0.35,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )


class AgentState(TypedDict, total=False):
    """LangGraph 所有节点共享的状态结构。

    使用 total=False 是因为每个节点只返回自己修改的字段，
    不需要重复返回完整状态。
    """

    question_history: Annotated[list[str], add]

    original_question: str
    retrieval_query: str

    top_k: int
    filters: dict[str, str]

    retrieval_attempts: int
    max_retrieval_attempts: int
    min_relevance_score: float

    retrieval_results: list[SearchResult]
    answer: EvidenceAnswer | None

    route: AgentRoute
    failure_reason: str | None
    error_type: str | None


def create_initial_agent_state(
    request: KnowledgeAgentRequest,
) -> AgentState:
    """把经过校验的请求转换为 LangGraph 初始状态。"""

    normalized_question = request.question

    return AgentState(
        question_history=[normalized_question],
        original_question=normalized_question,
        retrieval_query=normalized_question,
        top_k=request.top_k,
        filters=request.filters.copy(),
        retrieval_attempts=0,
        max_retrieval_attempts=request.max_retrieval_attempts,
        min_relevance_score=request.min_relevance_score,
        retrieval_results=[],
        answer=None,
        route=AgentRoute.RETRIEVE,
        failure_reason=None,
        error_type=None,
    )
