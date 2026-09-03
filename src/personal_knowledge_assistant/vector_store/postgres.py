from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from math import fsum, isfinite, sqrt
from typing import Any
from uuid import UUID

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentListResult,
    DocumentStatus,
    DocumentType,
    SearchResult,
    StoredDocument,
)
from personal_knowledge_assistant.vector_store.exceptions import (
    InvalidVectorError,
    UnsupportedVectorFilterError,
    VectorCountMismatchError,
    VectorDimensionMismatchError,
    VectorStoreError,
)


class PgVectorStore:
    """使用PostgreSQL和pgvector实现的向量存储。"""

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
        database_url: str,
        schema: str,
        dimension: int,
        pool_min_size: int,
        pool_max_size: int,
        connect_timeout_seconds: int,
        tenant_id: str = "local",
        tenant_id_provider: Callable[[], str] | None = None,
    ) -> None:
        if not database_url.strip():
            raise ValueError("Database URL cannot be empty.")

        if not schema.strip():
            raise ValueError("Database schema cannot be empty.")

        if dimension <= 0:
            raise ValueError("Vector dimension must be greater than zero.")

        if pool_min_size < 0:
            raise ValueError("Pool minimum size cannot be negative.")

        if pool_max_size < 1:
            raise ValueError("Pool maximum size must be greater than zero.")

        if pool_max_size < pool_min_size:
            raise ValueError("Pool maximum size cannot be smaller than minimum size.")

        normalized_tenant_id = tenant_id.strip()

        if not normalized_tenant_id:
            raise ValueError("Tenant ID cannot be empty.")

        self._schema = schema
        self._dimension = dimension
        self._default_tenant_id = normalized_tenant_id
        self._tenant_id_provider = tenant_id_provider
        self._connect_timeout_seconds = connect_timeout_seconds

        self._pool = AsyncConnectionPool[psycopg.AsyncConnection[Any]](
            conninfo=database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            open=False,
            timeout=float(connect_timeout_seconds),
            kwargs={
                "connect_timeout": connect_timeout_seconds,
            },
            configure=self._configure_connection,
            name="pka-pgvector",
        )

    @property
    def dimension(self) -> int:
        """返回向量维度。"""

        return self._dimension

    @property
    def tenant_id(self) -> str:
        """返回当前请求租户；没有请求上下文时返回默认租户。"""

        tenant_id = (
            self._tenant_id_provider()
            if self._tenant_id_provider is not None
            else self._default_tenant_id
        )
        normalized_tenant_id = tenant_id.strip()

        if not normalized_tenant_id:
            raise VectorStoreError("Tenant ID provider returned an empty tenant ID.")

        return normalized_tenant_id

    @staticmethod
    async def _configure_connection(
        connection: psycopg.AsyncConnection[Any],
    ) -> None:
        """为连接注册pgvector数据类型。"""

        await register_vector_async(connection)

    async def open(self) -> None:
        """打开连接池并验证数据库向量维度。"""

        try:
            await self._pool.open(
                wait=True,
                timeout=float(self._connect_timeout_seconds),
            )

            await self._verify_database_dimension()
        except VectorStoreError:
            await self._pool.close()
            raise
        except psycopg.Error as exc:
            await self._pool.close()

            raise VectorStoreError("Failed to open PostgreSQL vector store.") from exc

    async def close(self) -> None:
        """关闭连接池。"""

        await self._pool.close()

    async def _verify_database_dimension(self) -> None:
        """确认数据库向量列与Embedding维度一致。"""

        table_name = f"{self._schema}.document_chunks"

        query = """
            SELECT format_type(
                attribute.atttypid,
                attribute.atttypmod
            )
            FROM pg_attribute AS attribute
            WHERE attribute.attrelid = %s::regclass
              AND attribute.attname = 'embedding'
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        """

        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                query,
                (table_name,),
            )
            row = await cursor.fetchone()

        if row is None:
            raise VectorStoreError("document_chunks.embedding does not exist.")

        expected_type = f"vector({self._dimension})"
        actual_type = str(row[0])

        if actual_type != expected_type:
            raise VectorDimensionMismatchError(
                f"Database expects {actual_type}, but settings require {expected_type}."
            )

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """批量新增或更新文档片段。"""

        prepared = self._prepare_batch(
            chunks,
            embeddings,
        )

        if not prepared:
            return

        document_query = sql.SQL(
            """
            INSERT INTO {}.documents (
                tenant_id,
                document_id,
                source_path,
                file_name,
                document_type,
                title,
                status
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                NULL,
                'ready'
            )
            ON CONFLICT (
                tenant_id,
                document_id
            )
            DO UPDATE SET
                source_path = EXCLUDED.source_path,
                file_name = EXCLUDED.file_name,
                document_type = EXCLUDED.document_type,
                status = 'ready',
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """
        ).format(sql.Identifier(self._schema))

        version_query = sql.SQL(
            """
            INSERT INTO {}.document_versions (
                document_key,
                content_hash,
                active,
                metadata,
                modified_at
            )
            VALUES (
                %s,
                %s,
                TRUE,
                %s,
                %s
            )
            ON CONFLICT (
                document_key,
                content_hash
            )
            DO UPDATE SET
                active = TRUE,
                metadata = EXCLUDED.metadata,
                modified_at = EXCLUDED.modified_at,
                deactivated_at = NULL
            RETURNING id
            """
        ).format(sql.Identifier(self._schema))

        chunk_query = sql.SQL(
            """
            INSERT INTO {}.document_chunks (
                tenant_id,
                document_key,
                document_version_id,
                document_id,
                chunk_id,
                chunk_index,
                content_hash,
                text_content,
                source_path,
                file_name,
                document_type,
                page_number,
                section_path,
                metadata,
                embedding,
                active,
                deactivated_at
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                TRUE,
                NULL
            )
            ON CONFLICT (
                tenant_id,
                chunk_id
            )
            DO UPDATE SET
                document_key = EXCLUDED.document_key,
                document_version_id = EXCLUDED.document_version_id,
                document_id = EXCLUDED.document_id,
                chunk_index = EXCLUDED.chunk_index,
                content_hash = EXCLUDED.content_hash,
                text_content = EXCLUDED.text_content,
                source_path = EXCLUDED.source_path,
                file_name = EXCLUDED.file_name,
                document_type = EXCLUDED.document_type,
                page_number = EXCLUDED.page_number,
                section_path = EXCLUDED.section_path,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                active = TRUE,
                deactivated_at = NULL
            """
        ).format(sql.Identifier(self._schema))

        try:
            async with self._pool.connection() as connection:
                for chunk, embedding in prepared:
                    document_cursor = await connection.execute(
                        document_query,
                        (
                            self.tenant_id,
                            chunk.document_id,
                            chunk.metadata.source_path,
                            chunk.metadata.file_name,
                            chunk.metadata.document_type.value,
                        ),
                    )
                    document_row = await document_cursor.fetchone()

                    if document_row is None:
                        raise VectorStoreError("Document upsert returned no ID.")

                    document_key = document_row[0]

                    version_cursor = await connection.execute(
                        version_query,
                        (
                            document_key,
                            chunk.metadata.content_hash,
                            Jsonb(
                                {
                                    "source_path": (chunk.metadata.source_path),
                                    "file_name": (chunk.metadata.file_name),
                                }
                            ),
                            chunk.metadata.modified_at,
                        ),
                    )
                    version_row = await version_cursor.fetchone()

                    if version_row is None:
                        raise VectorStoreError("Document version upsert returned no ID.")

                    document_version_id = version_row[0]

                    await connection.execute(
                        chunk_query,
                        (
                            self.tenant_id,
                            document_key,
                            document_version_id,
                            chunk.document_id,
                            chunk.chunk_id,
                            chunk.metadata.chunk_index,
                            chunk.metadata.content_hash,
                            chunk.text,
                            chunk.metadata.source_path,
                            chunk.metadata.file_name,
                            chunk.metadata.document_type.value,
                            chunk.metadata.page_number,
                            Jsonb(chunk.metadata.section_path),
                            Jsonb(
                                {
                                    "part_index": (chunk.metadata.part_index),
                                    "chunk_index": (chunk.metadata.chunk_index),
                                }
                            ),
                            Vector(list(embedding)),
                        ),
                    )
        except VectorStoreError:
            raise
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL vector upsert failed.") from exc

    def _prepare_batch(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> list[
        tuple[
            DocumentChunk,
            tuple[float, ...],
        ]
    ]:
        """先验证整个批次，避免写入一半。"""

        if len(chunks) != len(embeddings):
            raise VectorCountMismatchError(
                f"Received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        prepared: list[
            tuple[
                DocumentChunk,
                tuple[float, ...],
            ]
        ] = []

        seen_chunk_ids: set[str] = set()
        document_hashes: dict[str, str] = {}

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            if chunk.chunk_id in seen_chunk_ids:
                raise VectorStoreError(f"Duplicate chunk ID in one batch: {chunk.chunk_id}")

            seen_chunk_ids.add(chunk.chunk_id)

            existing_hash = document_hashes.get(chunk.document_id)

            if existing_hash is not None and existing_hash != chunk.metadata.content_hash:
                raise VectorStoreError(
                    "One batch cannot contain multiple versions of the same document."
                )

            document_hashes[chunk.document_id] = chunk.metadata.content_hash

            prepared.append(
                (
                    chunk.model_copy(deep=True),
                    self._validate_vector(embedding),
                )
            )

        return prepared

    def _validate_vector(
        self,
        embedding: Sequence[float],
    ) -> tuple[float, ...]:
        """验证向量维度和数值。"""

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

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """使用余弦距离检索活动文档片段。"""

        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")

        normalized_filters = filters or {}

        self._validate_filters(normalized_filters)

        query_vector = self._validate_vector(query_embedding)

        filter_columns = {
            "document_id": sql.SQL("chunk.document_id"),
            "content_hash": sql.SQL("chunk.content_hash"),
            "source_path": sql.SQL("chunk.source_path"),
            "file_name": sql.SQL("chunk.file_name"),
            "document_type": sql.SQL("chunk.document_type"),
            "page_number": sql.SQL("chunk.page_number::text"),
        }

        where_parts: list[sql.Composable] = [
            sql.SQL("chunk.tenant_id = %s"),
            sql.SQL("chunk.active = TRUE"),
            sql.SQL("version.active = TRUE"),
        ]

        parameters: list[Any] = [
            Vector(list(query_vector)),
            self.tenant_id,
        ]

        for name, expected in normalized_filters.items():
            where_parts.append(sql.SQL("{} = %s").format(filter_columns[name]))
            parameters.append(expected)

        parameters.append(limit)

        query = sql.SQL(
            """
            WITH query_vector(embedding) AS (
                VALUES (%s::vector)
            )
            SELECT
                chunk.chunk_id,
                chunk.document_id,
                chunk.text_content,
                chunk.content_hash,
                chunk.source_path,
                chunk.file_name,
                chunk.document_type,
                version.modified_at,
                COALESCE(
                    (
                        chunk.metadata
                        ->> 'part_index'
                    )::integer,
                    0
                ),
                chunk.chunk_index,
                chunk.page_number,
                chunk.section_path,
                chunk.embedding
                    <=> query_vector.embedding
                    AS cosine_distance
            FROM {}.document_chunks AS chunk
            INNER JOIN {}.document_versions AS version
                ON version.id = chunk.document_version_id
            CROSS JOIN query_vector
            WHERE {}
            ORDER BY
                cosine_distance ASC,
                chunk.chunk_id ASC
            LIMIT %s
            """
        ).format(
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
            sql.SQL(" AND ").join(where_parts),
        )

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    query,
                    parameters,
                )
                rows = await cursor.fetchall()
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL vector search failed.") from exc

        results: list[SearchResult] = []

        for row in rows:
            modified_at = row[7]

            if not isinstance(
                modified_at,
                datetime,
            ):
                raise VectorStoreError("Stored document version has no modification timestamp.")

            section_path_value = row[11]

            if not isinstance(
                section_path_value,
                list,
            ):
                raise VectorStoreError("Stored section_path is invalid.")

            cosine_distance = float(row[12])
            score = 1.0 - cosine_distance
            score = max(
                -1.0,
                min(1.0, score),
            )

            results.append(
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id=str(row[0]),
                        document_id=str(row[1]),
                        text=str(row[2]),
                        metadata=ChunkMetadata(
                            content_hash=str(row[3]),
                            source_path=str(row[4]),
                            file_name=str(row[5]),
                            document_type=(DocumentType(str(row[6]))),
                            modified_at=modified_at,
                            part_index=int(row[8]),
                            chunk_index=int(row[9]),
                            page_number=(int(row[10]) if row[10] is not None else None),
                            section_path=[str(value) for value in section_path_value],
                        ),
                    ),
                    score=score,
                )
            )

        return results

    def _validate_filters(
        self,
        filters: dict[str, str],
    ) -> None:
        """拒绝未定义的过滤条件。"""

        unsupported = set(filters) - self._SUPPORTED_FILTERS

        if unsupported:
            names = ", ".join(sorted(unsupported))

            raise UnsupportedVectorFilterError(f"Unsupported filters: {names}")

    async def deactivate_document(
        self,
        document_id: str,
    ) -> int:
        """停用指定逻辑文档的活动Chunk和版本。"""

        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise ValueError("Document ID cannot be empty.")

        chunk_query = sql.SQL(
            """
            UPDATE {}.document_chunks
            SET
                active = FALSE,
                deactivated_at = CURRENT_TIMESTAMP
            WHERE tenant_id = %s
              AND document_id = %s
              AND active = TRUE
            """
        ).format(sql.Identifier(self._schema))

        version_query = sql.SQL(
            """
            UPDATE {}.document_versions
            SET
                active = FALSE,
                deactivated_at = CURRENT_TIMESTAMP
            WHERE active = TRUE
              AND document_key IN (
                  SELECT id
                  FROM {}.documents
                  WHERE tenant_id = %s
                    AND document_id = %s
              )
            """
        ).format(
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
        )

        try:
            async with self._pool.connection() as connection:
                chunk_cursor = await connection.execute(
                    chunk_query,
                    (
                        self.tenant_id,
                        normalized_document_id,
                    ),
                )

                affected = chunk_cursor.rowcount

                await connection.execute(
                    version_query,
                    (
                        self.tenant_id,
                        normalized_document_id,
                    ),
                )
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL document deactivation failed.") from exc

        return max(0, affected)

    async def get_document_by_id(
        self,
        document_id: str,
    ) -> StoredDocument | None:
        """根据稳定的document_id查询文档。"""

        normalized_document_id = document_id.strip()

        if not normalized_document_id:
            raise ValueError("Document ID cannot be empty.")

        return await self._fetch_document(
            where_clause=sql.SQL("document.document_id = %s"),
            value=normalized_document_id,
        )

    async def get_document(
        self,
        document_key: UUID,
    ) -> StoredDocument | None:
        """根据数据库UUID主键查询文档。"""

        return await self._fetch_document(
            where_clause=sql.SQL("document.id = %s"),
            value=document_key,
        )

    async def _fetch_document(
        self,
        *,
        where_clause: sql.Composable,
        value: str | UUID,
    ) -> StoredDocument | None:
        """执行单个文档查询。"""

        query = sql.SQL(
            """
            SELECT
                document.id,
                document.tenant_id,
                document.document_id,
                document.source_path,
                document.file_name,
                document.document_type,
                document.title,
                document.status,
                version.content_hash,
                COUNT(chunk.id) FILTER (
                    WHERE chunk.active = TRUE
                ) AS active_chunk_count,
                document.created_at,
                document.updated_at
            FROM {}.documents AS document
            LEFT JOIN {}.document_versions AS version
                ON version.document_key = document.id
               AND version.active = TRUE
            LEFT JOIN {}.document_chunks AS chunk
                ON chunk.document_key = document.id
               AND chunk.active = TRUE
            WHERE document.tenant_id = %s
              AND {}
            GROUP BY
                document.id,
                version.content_hash
            """
        ).format(
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
            where_clause,
        )

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    query,
                    (
                        self.tenant_id,
                        value,
                    ),
                )
                row = await cursor.fetchone()
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL document query failed.") from exc

        return self._document_from_row(row)

    async def list_documents(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool = False,
    ) -> DocumentListResult:
        """分页查询当前租户的文档。"""

        if limit < 1:
            raise ValueError("Document list limit must be greater than zero.")

        if limit > 100:
            raise ValueError("Document list limit cannot exceed 100.")

        if offset < 0:
            raise ValueError("Document list offset cannot be negative.")

        where_parts: list[sql.Composable] = [
            sql.SQL("document.tenant_id = %s"),
        ]

        if not include_deleted:
            where_parts.append(sql.SQL("document.status <> 'deleted'"))

        where_clause = sql.SQL(" AND ").join(where_parts)

        list_query = sql.SQL(
            """
            SELECT
                document.id,
                document.tenant_id,
                document.document_id,
                document.source_path,
                document.file_name,
                document.document_type,
                document.title,
                document.status,
                version.content_hash,
                COUNT(chunk.id) FILTER (
                    WHERE chunk.active = TRUE
                ) AS active_chunk_count,
                document.created_at,
                document.updated_at
            FROM {}.documents AS document
            LEFT JOIN {}.document_versions AS version
                ON version.document_key = document.id
               AND version.active = TRUE
            LEFT JOIN {}.document_chunks AS chunk
                ON chunk.document_key = document.id
               AND chunk.active = TRUE
            WHERE {}
            GROUP BY
                document.id,
                version.content_hash
            ORDER BY
                document.updated_at DESC,
                document.id ASC
            LIMIT %s
            OFFSET %s
            """
        ).format(
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
            sql.Identifier(self._schema),
            where_clause,
        )

        count_query = sql.SQL(
            """
            SELECT COUNT(*)
            FROM {}.documents AS document
            WHERE {}
            """
        ).format(
            sql.Identifier(self._schema),
            where_clause,
        )

        try:
            async with self._pool.connection() as connection:
                list_cursor = await connection.execute(
                    list_query,
                    (
                        self.tenant_id,
                        limit,
                        offset,
                    ),
                )
                rows = await list_cursor.fetchall()

                count_cursor = await connection.execute(
                    count_query,
                    (self.tenant_id,),
                )
                count_row = await count_cursor.fetchone()
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL document list query failed.") from exc

        if count_row is None:
            raise VectorStoreError("PostgreSQL document count returned no result.")

        return DocumentListResult(
            items=[self._document_from_required_row(row) for row in rows],
            total=int(count_row[0]),
            limit=limit,
            offset=offset,
        )

    async def delete_document(
        self,
        document_key: UUID,
    ) -> bool:
        """软删除文档并停用相关版本和Chunk。"""

        document_query = sql.SQL(
            """
            UPDATE {}.documents
            SET
                status = 'deleted',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
              AND tenant_id = %s
              AND status <> 'deleted'
            """
        ).format(sql.Identifier(self._schema))

        version_query = sql.SQL(
            """
            UPDATE {}.document_versions
            SET
                active = FALSE,
                deactivated_at = CURRENT_TIMESTAMP
            WHERE document_key = %s
              AND active = TRUE
            """
        ).format(sql.Identifier(self._schema))

        chunk_query = sql.SQL(
            """
            UPDATE {}.document_chunks
            SET
                active = FALSE,
                deactivated_at = CURRENT_TIMESTAMP
            WHERE document_key = %s
              AND tenant_id = %s
              AND active = TRUE
            """
        ).format(sql.Identifier(self._schema))

        try:
            async with self._pool.connection() as connection:
                document_cursor = await connection.execute(
                    document_query,
                    (
                        document_key,
                        self.tenant_id,
                    ),
                )

                if document_cursor.rowcount <= 0:
                    return False

                await connection.execute(
                    version_query,
                    (document_key,),
                )

                await connection.execute(
                    chunk_query,
                    (
                        document_key,
                        self.tenant_id,
                    ),
                )
        except psycopg.Error as exc:
            raise VectorStoreError("PostgreSQL document deletion failed.") from exc

        return True

    def _document_from_row(
        self,
        row: Sequence[Any] | None,
    ) -> StoredDocument | None:
        """把数据库记录转换为领域模型。"""

        if row is None:
            return None

        return self._document_from_required_row(row)

    def _document_from_required_row(
        self,
        row: Sequence[Any],
    ) -> StoredDocument:
        """把非空数据库记录转换为领域模型。"""

        content_hash = str(row[8]).strip() if row[8] is not None else None

        return StoredDocument(
            document_key=row[0],
            tenant_id=str(row[1]),
            document_id=str(row[2]),
            source_path=str(row[3]),
            file_name=str(row[4]),
            document_type=DocumentType(str(row[5])),
            title=(str(row[6]) if row[6] is not None else None),
            status=DocumentStatus(str(row[7])),
            active_content_hash=content_hash,
            active_chunk_count=int(row[9]),
            created_at=row[10],
            updated_at=row[11],
        )
