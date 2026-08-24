from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentRequest,
)

NonEmptyApiText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]

ThreadId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
]


class HealthResponse(BaseModel):
    """服务健康检查响应。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    status: Literal["ok"]
    service: Literal["personal-knowledge-assistant"] = "personal-knowledge-assistant"


class AgentQueryRequest(BaseModel):
    """HTTP Agent 查询请求。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    thread_id: ThreadId
    question: NonEmptyApiText

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    filters: dict[
        NonEmptyApiText,
        NonEmptyApiText,
    ] = Field(
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

    def to_agent_request(
        self,
    ) -> KnowledgeAgentRequest:
        """转换为 Agent 领域请求。"""

        return KnowledgeAgentRequest(
            question=self.question,
            top_k=self.top_k,
            filters=self.filters.copy(),
            max_retrieval_attempts=(self.max_retrieval_attempts),
            min_relevance_score=(self.min_relevance_score),
        )


class AgentHistoryResponse(BaseModel):
    """指定 Agent 会话的问题历史。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    thread_id: ThreadId
    questions: list[NonEmptyApiText] = Field(
        default_factory=list,
    )


class ApiErrorResponse(BaseModel):
    """不暴露内部异常内容的错误响应。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    detail: NonEmptyApiText
