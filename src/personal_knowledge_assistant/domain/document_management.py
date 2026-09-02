from enum import StrEnum
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
)

from personal_knowledge_assistant.domain.documents import (
    DocumentType,
    Sha256Hash,
)


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class StoredDocument(BaseModel):
    """数据库中的逻辑文档。"""

    document_key: UUID
    tenant_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    document_type: DocumentType
    title: str | None = None
    status: DocumentStatus
    active_content_hash: Sha256Hash | None = None
    active_chunk_count: int = Field(default=0, ge=0)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class DocumentListResult(BaseModel):
    """分页文档列表。"""

    items: list[StoredDocument]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
