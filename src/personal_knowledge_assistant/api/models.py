from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentRequest,
)
from personal_knowledge_assistant.domain import (
    DocumentStatus,
    DocumentType,
    ImportStatus,
)
from personal_knowledge_assistant.indexing import (
    IndexingStatus,
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


class DocumentResponse(BaseModel):
    """单个持久化文档的当前状态。"""

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    document_key: UUID
    document_id: NonEmptyApiText
    file_name: NonEmptyApiText
    document_type: DocumentType
    status: DocumentStatus
    active_content_hash: str | None = None
    active_chunk_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """分页文档列表响应。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    items: list[DocumentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


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


class DocumentImportResponse(BaseModel):
    """文档上传和索引结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    document_id: NonEmptyApiText
    file_name: NonEmptyApiText
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    import_status: ImportStatus
    indexing_status: IndexingStatus
    chunk_count: int = Field(ge=0)
    deactivated_chunk_count: int = Field(
        ge=0,
    )
