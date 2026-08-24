from enum import StrEnum

from pydantic import BaseModel, Field

from personal_knowledge_assistant.domain.documents import (
    Sha256Hash,
)


class IndexingStatus(StrEnum):
    """文档索引执行状态。"""

    INDEXED = "indexed"
    SKIPPED = "skipped"


class IndexingResult(BaseModel):
    """一次文档索引的执行结果。"""

    status: IndexingStatus
    document_id: str = Field(min_length=1)
    content_hash: Sha256Hash
    chunk_count: int = Field(ge=0)
    deactivated_chunk_count: int = Field(
        default=0,
        ge=0,
    )
