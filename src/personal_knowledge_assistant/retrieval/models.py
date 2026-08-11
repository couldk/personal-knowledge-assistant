from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalTraceStatus(StrEnum):
    """检索 Trace 执行状态。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RetrievalHitTrace(BaseModel):
    """一条检索命中的 Trace。"""

    chunk_id: str = Field(min_length=1)
    score: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )


class RetrievalTrace(BaseModel):
    """一次检索调用的可观测数据。"""

    trace_id: UUID
    status: RetrievalTraceStatus

    # 查询文本的 SHA-256，不保存原始问题。
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    top_k: int = Field(ge=1)
    result_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)

    embedding_model: str = Field(min_length=1)

    hits: list[RetrievalHitTrace] = Field(default_factory=list)

    error_type: str | None = None
