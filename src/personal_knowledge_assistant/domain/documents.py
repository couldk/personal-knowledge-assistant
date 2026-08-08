from enum import StrEnum
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    StringConstraints,
)

Sha256Hash = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class DocumentType(StrEnum):
    """当前支持的文档类型。"""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"


class ImportStatus(StrEnum):
    """文档导入状态。"""

    CREATED = "created"
    UNCHANGED = "unchanged"
    UPDATED = "updated"


class DocumentMetadata(BaseModel):
    """文件级元数据。"""

    source_path: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    document_type: DocumentType
    file_size: int = Field(ge=0)
    modified_at: AwareDatetime
    title: str | None = Field(default=None, min_length=1)
    extra: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class DocumentPart(BaseModel):
    """文档中可以独立定位的内容单元。"""

    part_index: int = Field(ge=0)
    text: str = Field(min_length=1)

    # PDF使用，从1开始；Markdown和TXT通常为None。
    page_number: int | None = Field(default=None, ge=1)

    # Markdown使用，例如 ["LangGraph", "Memory"]。
    section_path: list[str] = Field(default_factory=list)


class LoadedDocument(BaseModel):
    """Loader解析完成后的统一文档对象。"""

    document_id: str = Field(min_length=1)
    content_hash: Sha256Hash
    metadata: DocumentMetadata
    parts: list[DocumentPart] = Field(min_length=1)

    @property
    def text(self) -> str:
        """把所有内容单元合并为完整文本。"""

        return "\n\n".join(part.text for part in self.parts)


class ImportResult(BaseModel):
    """一次文档导入操作的结果。"""

    status: ImportStatus
    document: LoadedDocument
    previous_content_hash: Sha256Hash | None = None
