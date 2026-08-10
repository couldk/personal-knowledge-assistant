from typing import Protocol, runtime_checkable

from personal_knowledge_assistant.domain import (
    DocumentChunk,
    LoadedDocument,
)


@runtime_checkable
class DocumentChunker(Protocol):
    """把一个已加载文档转换为可检索Chunk。"""

    def split(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """切分文档并返回带来源信息的Chunk。"""

        ...
