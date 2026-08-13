from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from personal_knowledge_assistant.domain.documents import (
    Sha256Hash,
)
from personal_knowledge_assistant.domain.retrieval import (
    ChunkId,
)


class AnsweringTraceStatus(StrEnum):
    """回答流程执行状态。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnsweringTrace(BaseModel):
    """一次回答生成操作的安全 Trace。"""

    trace_id: UUID
    status: AnsweringTraceStatus
    question_hash: Sha256Hash

    duration_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    model: str | None = Field(
        default=None,
        min_length=1,
    )

    refused: bool | None = None

    citation_ids: list[ChunkId] = Field(
        default_factory=list,
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
    )

    error_type: str | None = Field(
        default=None,
        min_length=1,
    )
