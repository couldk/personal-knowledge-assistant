import json
from datetime import datetime
from hashlib import sha256
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

from personal_knowledge_assistant.domain.documents import (
    DocumentType,
    Sha256Hash,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]

ChunkId = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


def create_chunk_id(
    *,
    document_id: str,
    content_hash: str,
    part_index: int,
    chunk_index: int,
) -> str:
    """根据文档版本和片段位置生成确定性的Chunk ID。"""

    if not document_id.strip():
        raise ValueError("document_id cannot be empty.")

    if part_index < 0:
        raise ValueError("part_index cannot be negative.")

    if chunk_index < 0:
        raise ValueError("chunk_index cannot be negative.")

    identity = {
        "document_id": document_id,
        "content_hash": content_hash,
        "part_index": part_index,
        "chunk_index": chunk_index,
    }

    payload = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256(payload).hexdigest()


class ChunkMetadata(BaseModel):
    """一个Chunk对应的来源和版本信息。"""

    content_hash: Sha256Hash
    source_path: NonEmptyString
    file_name: NonEmptyString
    document_type: DocumentType
    modified_at: datetime

    part_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)

    page_number: int | None = Field(
        default=None,
        ge=1,
    )
    section_path: list[NonEmptyString] = Field(
        default_factory=list,
    )


class DocumentChunk(BaseModel):
    """进入向量索引的最小检索单元。"""

    chunk_id: ChunkId
    document_id: NonEmptyString
    text: NonEmptyString
    metadata: ChunkMetadata


class SearchResult(BaseModel):
    """一次向量检索返回的Chunk和相似度分数。"""

    chunk: DocumentChunk
    score: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )


class ChunkingConfig(BaseModel):
    """文档切块参数，单位为Token。"""

    chunk_size: int = Field(
        default=512,
        ge=64,
        le=8192,
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=2048,
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        """重叠长度必须小于Chunk长度。"""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        return self


class RetrievalQuery(BaseModel):
    """Top-k检索请求。"""

    query: NonEmptyString
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    filters: dict[NonEmptyString, NonEmptyString] = Field(
        default_factory=dict,
    )
