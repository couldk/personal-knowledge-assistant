from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite, sqrt

from personal_knowledge_assistant.domain import (
    DocumentChunk,
    SearchResult,
)
from personal_knowledge_assistant.vector_store.exceptions import (
    InvalidVectorError,
    UnsupportedVectorFilterError,
    VectorCountMismatchError,
    VectorDimensionMismatchError,
    VectorStoreError,
)


@dataclass(slots=True)
class _StoredVector:
    """内存向量库中的一条记录。"""

    chunk: DocumentChunk
    embedding: tuple[float, ...]
    active: bool = True


class InMemoryVectorStore:
    """使用余弦相似度实现的内存向量库。"""

    _SUPPORTED_FILTERS = frozenset(
        {
            "document_id",
            "content_hash",
            "source_path",
            "file_name",
            "document_type",
            "page_number",
        }
    )

    def __init__(
        self,
        *,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Vector dimension must be greater than zero.")

        self._dimension = dimension
        self._records: dict[str, _StoredVector] = {}

    @property
    def dimension(self) -> int:
        """返回向量库要求的向量维度。"""

        return self._dimension

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """新增或者覆盖 Chunk 向量。"""

        if len(chunks) != len(embeddings):
            raise VectorCountMismatchError(
                f"Received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        prepared: list[tuple[DocumentChunk, tuple[float, ...]]] = []
        seen_chunk_ids: set[str] = set()

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            if chunk.chunk_id in seen_chunk_ids:
                raise VectorStoreError(f"Duplicate chunk ID in one batch: {chunk.chunk_id}")

            seen_chunk_ids.add(chunk.chunk_id)

            prepared.append(
                (
                    chunk.model_copy(deep=True),
                    self._validate_vector(embedding),
                )
            )

        # 所有数据验证成功后再修改存储，避免写入一半。
        for chunk, embedding in prepared:
            self._records[chunk.chunk_id] = _StoredVector(
                chunk=chunk,
                embedding=embedding,
                active=True,
            )

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """按照余弦相似度搜索 top-k Chunk。"""

        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")

        normalized_filters = filters or {}
        self._validate_filters(normalized_filters)

        query_vector = self._validate_vector(
            query_embedding,
        )

        results: list[SearchResult] = []

        for record in self._records.values():
            if not record.active:
                continue

            if not self._matches_filters(
                record.chunk,
                normalized_filters,
            ):
                continue

            score = self._cosine_similarity(
                query_vector,
                record.embedding,
            )

            results.append(
                SearchResult(
                    chunk=record.chunk.model_copy(deep=True),
                    score=score,
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.chunk_id,
            )
        )

        return results[:limit]

    async def deactivate_document(
        self,
        document_id: str,
    ) -> int:
        """停用指定文档对应的全部有效 Chunk。"""

        affected = 0

        for record in self._records.values():
            if record.active and record.chunk.document_id == document_id:
                record.active = False
                affected += 1

        return affected

    def _validate_vector(
        self,
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        vector = tuple(float(value) for value in embedding)

        if len(vector) != self._dimension:
            raise VectorDimensionMismatchError(
                f"Expected {self._dimension} dimensions, received {len(vector)}."
            )

        if not all(isfinite(value) for value in vector):
            raise InvalidVectorError("Vector contains NaN or infinity.")

        norm = sqrt(fsum(value * value for value in vector))

        if norm == 0:
            raise InvalidVectorError("Vector must not be a zero vector.")

        return vector

    def _validate_filters(
        self,
        filters: dict[str, str],
    ) -> None:
        unsupported = set(filters) - self._SUPPORTED_FILTERS

        if unsupported:
            names = ", ".join(sorted(unsupported))

            raise UnsupportedVectorFilterError(f"Unsupported filters: {names}")

    @staticmethod
    def _matches_filters(
        chunk: DocumentChunk,
        filters: dict[str, str],
    ) -> bool:
        page_number = (
            str(chunk.metadata.page_number) if chunk.metadata.page_number is not None else ""
        )

        values = {
            "document_id": chunk.document_id,
            "content_hash": chunk.metadata.content_hash,
            "source_path": chunk.metadata.source_path,
            "file_name": chunk.metadata.file_name,
            "document_type": (chunk.metadata.document_type.value),
            "page_number": page_number,
        }

        return all(values[name] == expected for name, expected in filters.items())

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        dot_product = fsum(
            left_value * right_value
            for left_value, right_value in zip(
                left,
                right,
                strict=True,
            )
        )

        left_norm = sqrt(fsum(value * value for value in left))
        right_norm = sqrt(fsum(value * value for value in right))

        score = dot_product / (left_norm * right_norm)

        # 防止浮点误差产生 1.0000000002。
        return max(-1.0, min(1.0, score))
