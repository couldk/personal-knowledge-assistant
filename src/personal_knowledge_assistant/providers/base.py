from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    ChatResponse,
    DocumentChunk,
    SearchResult,
)


@runtime_checkable
class ChatProvider(Protocol):
    """聊天模型统一接口。"""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
    ) -> ChatResponse:
        """根据消息生成回答。"""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding 模型统一接口。"""

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """批量生成文本向量。"""
        ...

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """为查询生成向量。"""
        ...


@runtime_checkable
class VectorStoreProvider(Protocol):
    """向量数据库统一接口。"""

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """新增或更新文档片段。"""
        ...

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """搜索相似片段。"""
        ...

    async def deactivate_document(
        self,
        document_id: str,
    ) -> int:
        """停用指定文档的旧片段，返回受影响数量。"""
        ...
