from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql

from personal_knowledge_assistant.application import (
    DocumentImportService,
)
from personal_knowledge_assistant.chunking import SentenceChunker
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    create_chunk_id,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
    IndexingStatus,
)
from personal_knowledge_assistant.ingestion import (
    create_default_ingestion_service,
)
from personal_knowledge_assistant.vector_store.postgres import (
    PgVectorStore,
)

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("PKA_RUN_POSTGRES_TESTS") != "1",
        reason=("Set PKA_RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests."),
    ),
]

DIMENSION = 1024


class IntegrationEmbeddingProvider:
    """为真实 pgvector 验收提供确定性的 1024 维向量。"""

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        return self._embed(query)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        return make_vector(0 if "vector" in lowered else 1)


def make_vector(
    primary_index: int,
    secondary_index: int | None = None,
) -> list[float]:
    vector = [0.0] * DIMENSION
    vector[primary_index] = 1.0

    if secondary_index is not None:
        vector[secondary_index] = 1.0

    return vector


def make_chunk(
    *,
    document_id: str,
    content_hash: str,
    chunk_index: int,
    text: str,
    file_name: str = "notes.txt",
    document_type: DocumentType = DocumentType.TEXT,
    page_number: int | None = None,
) -> DocumentChunk:
    chunk_id = create_chunk_id(
        document_id=document_id,
        content_hash=content_hash,
        part_index=0,
        chunk_index=chunk_index,
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata=ChunkMetadata(
            content_hash=content_hash,
            source_path=(f"D:/documents/{file_name}"),
            file_name=file_name,
            document_type=document_type,
            modified_at=datetime(
                2026,
                8,
                31,
                12,
                0,
                tzinfo=UTC,
            ),
            part_index=0,
            chunk_index=chunk_index,
            page_number=page_number,
            section_path=[],
        ),
    )


@pytest_asyncio.fixture
async def pg_store() -> AsyncIterator[PgVectorStore]:
    settings = Settings()
    tenant_id = f"test-{uuid4().hex}"

    store = PgVectorStore(
        database_url=settings.database_url,
        schema=settings.database_schema,
        dimension=DIMENSION,
        pool_min_size=1,
        pool_max_size=3,
        connect_timeout_seconds=10,
        tenant_id=tenant_id,
    )

    await store.open()

    try:
        yield store
    finally:
        delete_query = sql.SQL("DELETE FROM {}.documents WHERE tenant_id = %s").format(
            sql.Identifier(settings.database_schema)
        )

        async with await psycopg.AsyncConnection.connect(settings.database_url) as connection:
            await connection.execute(
                delete_query,
                (tenant_id,),
            )

        await store.close()


@pytest.mark.asyncio
async def test_pgvector_upsert_and_search(
    pg_store: PgVectorStore,
) -> None:
    document_id = "file:///integration/vector-search.txt"

    chunk_a = make_chunk(
        document_id=document_id,
        content_hash="a" * 64,
        chunk_index=0,
        text="Vector databases support semantic search.",
    )
    chunk_b = make_chunk(
        document_id=document_id,
        content_hash="a" * 64,
        chunk_index=1,
        text="Python is a programming language.",
    )
    chunk_c = make_chunk(
        document_id=document_id,
        content_hash="a" * 64,
        chunk_index=2,
        text="RAG combines retrieval and generation.",
    )

    await pg_store.upsert(
        [chunk_a, chunk_b, chunk_c],
        [
            make_vector(0),
            make_vector(1),
            make_vector(0, 1),
        ],
    )

    results = await pg_store.search(
        make_vector(0),
        limit=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        chunk_a.chunk_id,
        chunk_c.chunk_id,
        chunk_b.chunk_id,
    ]

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)
    assert results[2].score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_pgvector_search_applies_filters(
    pg_store: PgVectorStore,
) -> None:
    text_chunk = make_chunk(
        document_id="file:///integration/notes.txt",
        content_hash="b" * 64,
        chunk_index=0,
        text="Text document.",
        file_name="notes.txt",
    )
    markdown_chunk = make_chunk(
        document_id="file:///integration/guide.md",
        content_hash="c" * 64,
        chunk_index=0,
        text="Markdown document.",
        file_name="guide.md",
        document_type=DocumentType.MARKDOWN,
    )

    await pg_store.upsert(
        [text_chunk, markdown_chunk],
        [
            make_vector(0),
            make_vector(0),
        ],
    )

    results = await pg_store.search(
        make_vector(0),
        limit=5,
        filters={
            "document_type": "markdown",
        },
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == markdown_chunk.document_id


@pytest.mark.asyncio
async def test_pgvector_deactivate_document(
    pg_store: PgVectorStore,
) -> None:
    document_a = make_chunk(
        document_id="file:///integration/a.txt",
        content_hash="d" * 64,
        chunk_index=0,
        text="Document A.",
    )
    document_b = make_chunk(
        document_id="file:///integration/b.txt",
        content_hash="e" * 64,
        chunk_index=0,
        text="Document B.",
    )

    await pg_store.upsert(
        [document_a, document_b],
        [
            make_vector(0),
            make_vector(1),
        ],
    )

    affected = await pg_store.deactivate_document(document_a.document_id)
    second_affected = await pg_store.deactivate_document(document_a.document_id)

    results = await pg_store.search(
        make_vector(0),
        limit=5,
    )

    assert affected == 1
    assert second_affected == 0

    assert all(result.chunk.document_id != document_a.document_id for result in results)


@pytest.mark.asyncio
async def test_document_upload_is_indexed_and_searchable_in_pgvector(
    pg_store: PgVectorStore,
    tmp_path: Path,
) -> None:
    embedding_provider = IntegrationEmbeddingProvider()
    indexing_service = DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=64,
                chunk_overlap=8,
            )
        ),
        embedding_provider=embedding_provider,
        vector_store=pg_store,
    )
    import_service = DocumentImportService(
        ingestion_service=create_default_ingestion_service(),
        indexing_service=indexing_service,
        upload_directory=tmp_path,
        max_upload_bytes=1024,
    )

    outcome = await import_service.import_upload(
        file_name="day10-vector.txt",
        content=b"Vector databases support semantic retrieval.",
    )
    results = await pg_store.search(
        await embedding_provider.embed_query("vector search"),
        limit=5,
    )

    assert outcome.indexing_status is IndexingStatus.INDEXED
    assert outcome.chunk_count > 0
    assert results
    assert results[0].chunk.document_id == outcome.document_id
    assert results[0].chunk.metadata.file_name == "day10-vector.txt"
