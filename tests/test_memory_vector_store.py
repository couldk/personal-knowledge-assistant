from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.auth import (
    AuthenticatedPrincipal,
    reset_current_principal,
    set_current_principal,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    create_chunk_id,
)
from personal_knowledge_assistant.providers.base import (
    VectorStoreProvider,
)
from personal_knowledge_assistant.vector_store.exceptions import (
    InvalidVectorError,
    UnsupportedVectorFilterError,
    VectorCountMismatchError,
    VectorDimensionMismatchError,
    VectorStoreError,
)
from personal_knowledge_assistant.vector_store.factory import (
    create_vector_store,
)
from personal_knowledge_assistant.vector_store.memory import (
    InMemoryVectorStore,
)
from personal_knowledge_assistant.vector_store.postgres import (
    PgVectorStore,
)


def _make_chunk(
    *,
    document_id: str = "document-a",
    content_hash: str = "a" * 64,
    part_index: int = 0,
    chunk_index: int = 0,
    text: str = "Test document chunk.",
    file_name: str = "notes.txt",
    source_path: str = "D:/documents/notes.txt",
    document_type: DocumentType = DocumentType.TEXT,
    page_number: int | None = None,
) -> DocumentChunk:
    """创建用于向量库测试的 DocumentChunk。"""

    chunk_id = create_chunk_id(
        document_id=document_id,
        content_hash=content_hash,
        part_index=part_index,
        chunk_index=chunk_index,
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata=ChunkMetadata(
            content_hash=content_hash,
            source_path=source_path,
            file_name=file_name,
            document_type=document_type,
            modified_at=datetime(
                2026,
                8,
                11,
                10,
                0,
                tzinfo=UTC,
            ),
            part_index=part_index,
            chunk_index=chunk_index,
            page_number=page_number,
            section_path=[],
        ),
    )


def test_store_requires_positive_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        InMemoryVectorStore(dimension=0)


@pytest.mark.asyncio
async def test_upsert_and_search_returns_most_similar_chunk() -> None:
    store = InMemoryVectorStore(dimension=3)

    chunk_a = _make_chunk(
        chunk_index=0,
        text="Vector databases support semantic search.",
    )
    chunk_b = _make_chunk(
        chunk_index=1,
        text="Python is a programming language.",
    )
    chunk_c = _make_chunk(
        chunk_index=2,
        text="RAG combines retrieval and generation.",
    )

    await store.upsert(
        [chunk_a, chunk_b, chunk_c],
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
    )

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        chunk_a.chunk_id,
        chunk_c.chunk_id,
        chunk_b.chunk_id,
    ]

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(
        2**-0.5,
    )
    assert results[2].score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_respects_limit() -> None:
    store = InMemoryVectorStore(dimension=3)

    chunks = [_make_chunk(chunk_index=index) for index in range(3)]

    await store.upsert(
        chunks,
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=2,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_search_rejects_invalid_limit() -> None:
    store = InMemoryVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        await store.search(
            [1.0, 0.0, 0.0],
            limit=0,
        )


@pytest.mark.asyncio
async def test_equal_scores_are_ordered_by_chunk_id() -> None:
    store = InMemoryVectorStore(dimension=3)

    chunk_a = _make_chunk(
        document_id="document-a",
        chunk_index=0,
    )
    chunk_b = _make_chunk(
        document_id="document-b",
        content_hash="b" * 64,
        chunk_index=0,
    )

    await store.upsert(
        [chunk_b, chunk_a],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=2,
    )

    actual_ids = [result.chunk.chunk_id for result in results]
    expected_ids = sorted(
        [
            chunk_a.chunk_id,
            chunk_b.chunk_id,
        ]
    )

    assert actual_ids == expected_ids


@pytest.mark.asyncio
async def test_upsert_rejects_count_mismatch() -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk()

    with pytest.raises(
        VectorCountMismatchError,
        match="1 chunks but 0 embeddings",
    ):
        await store.upsert(
            [chunk],
            [],
        )


@pytest.mark.asyncio
async def test_upsert_rejects_wrong_dimension() -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk()

    with pytest.raises(
        VectorDimensionMismatchError,
        match="Expected 3 dimensions",
    ):
        await store.upsert(
            [chunk],
            [[1.0, 0.0]],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_vector",
    [
        [float("nan"), 0.0, 1.0],
        [float("inf"), 0.0, 1.0],
        [float("-inf"), 0.0, 1.0],
    ],
)
async def test_upsert_rejects_non_finite_values(
    invalid_vector: list[float],
) -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk()

    with pytest.raises(
        InvalidVectorError,
        match="NaN or infinity",
    ):
        await store.upsert(
            [chunk],
            [invalid_vector],
        )


@pytest.mark.asyncio
async def test_upsert_rejects_zero_vector() -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk()

    with pytest.raises(
        InvalidVectorError,
        match="zero vector",
    ):
        await store.upsert(
            [chunk],
            [[0.0, 0.0, 0.0]],
        )


@pytest.mark.asyncio
async def test_search_rejects_zero_query_vector() -> None:
    store = InMemoryVectorStore(dimension=3)

    with pytest.raises(
        InvalidVectorError,
        match="zero vector",
    ):
        await store.search(
            [0.0, 0.0, 0.0],
            limit=5,
        )


@pytest.mark.asyncio
async def test_upsert_rejects_duplicate_chunk_ids() -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk()

    with pytest.raises(
        VectorStoreError,
        match="Duplicate chunk ID",
    ):
        await store.upsert(
            [chunk, chunk],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )


@pytest.mark.asyncio
async def test_failed_batch_does_not_write_partial_data() -> None:
    store = InMemoryVectorStore(dimension=3)

    valid_chunk = _make_chunk(
        chunk_index=0,
    )
    invalid_chunk = _make_chunk(
        chunk_index=1,
    )

    with pytest.raises(
        VectorDimensionMismatchError,
    ):
        await store.upsert(
            [valid_chunk, invalid_chunk],
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0],
            ],
        )

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=5,
    )

    assert results == []


@pytest.mark.asyncio
async def test_upsert_replaces_existing_chunk() -> None:
    store = InMemoryVectorStore(dimension=3)

    original_chunk = _make_chunk(
        text="Original text.",
    )
    updated_chunk = original_chunk.model_copy(
        update={
            "text": "Updated text.",
        }
    )

    await store.upsert(
        [original_chunk],
        [[1.0, 0.0, 0.0]],
    )

    await store.upsert(
        [updated_chunk],
        [[0.0, 1.0, 0.0]],
    )

    results = await store.search(
        [0.0, 1.0, 0.0],
        limit=5,
    )

    assert len(results) == 1
    assert results[0].chunk.text == "Updated text."
    assert results[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_deactivate_document_hides_its_chunks() -> None:
    store = InMemoryVectorStore(dimension=3)

    document_a_chunk_1 = _make_chunk(
        document_id="document-a",
        chunk_index=0,
    )
    document_a_chunk_2 = _make_chunk(
        document_id="document-a",
        chunk_index=1,
    )
    document_b_chunk = _make_chunk(
        document_id="document-b",
        content_hash="b" * 64,
        chunk_index=0,
        file_name="other.txt",
        source_path="D:/documents/other.txt",
    )

    await store.upsert(
        [
            document_a_chunk_1,
            document_a_chunk_2,
            document_b_chunk,
        ],
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    affected = await store.deactivate_document("document-a")
    second_affected = await store.deactivate_document("document-a")

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=5,
    )

    assert affected == 2
    assert second_affected == 0
    assert len(results) == 1
    assert results[0].chunk.document_id == "document-b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filters", "expected_document_id"),
    [
        (
            {"document_id": "document-a"},
            "document-a",
        ),
        (
            {"file_name": "other.md"},
            "document-b",
        ),
        (
            {"document_type": "markdown"},
            "document-b",
        ),
        (
            {"page_number": "2"},
            "document-b",
        ),
    ],
)
async def test_search_applies_supported_filters(
    filters: dict[str, str],
    expected_document_id: str,
) -> None:
    store = InMemoryVectorStore(dimension=3)

    chunk_a = _make_chunk(
        document_id="document-a",
        chunk_index=0,
    )
    chunk_b = _make_chunk(
        document_id="document-b",
        content_hash="b" * 64,
        chunk_index=0,
        file_name="other.md",
        source_path="D:/documents/other.md",
        document_type=DocumentType.MARKDOWN,
        page_number=2,
    )

    await store.upsert(
        [chunk_a, chunk_b],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    results = await store.search(
        [1.0, 0.0, 0.0],
        limit=5,
        filters=filters,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == expected_document_id


@pytest.mark.asyncio
async def test_search_rejects_unsupported_filter() -> None:
    store = InMemoryVectorStore(dimension=3)

    with pytest.raises(
        UnsupportedVectorFilterError,
        match="Unsupported filters: tenant_id",
    ):
        await store.search(
            [1.0, 0.0, 0.0],
            limit=5,
            filters={
                "tenant_id": "user-a",
            },
        )


@pytest.mark.asyncio
async def test_search_returns_defensive_chunk_copy() -> None:
    store = InMemoryVectorStore(dimension=3)
    chunk = _make_chunk(
        text="Original text.",
    )

    await store.upsert(
        [chunk],
        [[1.0, 0.0, 0.0]],
    )

    first_results = await store.search(
        [1.0, 0.0, 0.0],
        limit=1,
    )
    first_results[0].chunk.text = "Modified externally."

    second_results = await store.search(
        [1.0, 0.0, 0.0],
        limit=1,
    )

    assert second_results[0].chunk.text == "Original text."


def test_store_implements_vector_store_protocol() -> None:
    store = InMemoryVectorStore(dimension=3)

    assert isinstance(
        store,
        VectorStoreProvider,
    )


def test_factory_creates_memory_store() -> None:
    settings = Settings.model_validate(
        {
            "vector_store_provider": "memory",
            "embedding_dimension": 3,
        }
    )

    store = create_vector_store(settings)

    assert isinstance(
        store,
        InMemoryVectorStore,
    )
    assert store.dimension == 3


def test_factory_requires_embedding_dimension() -> None:
    settings = Settings.model_validate(
        {
            "vector_store_provider": "memory",
            "embedding_dimension": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimension is required",
    ):
        create_vector_store(settings)


def test_factory_creates_pgvector_store() -> None:
    settings = Settings.model_validate(
        {
            "vector_store_provider": "pgvector",
            "embedding_dimension": 1024,
            "database_url": ("postgresql://pka:test@localhost:5432/personal_knowledge"),
            "database_schema": "pka",
            "database_pool_min_size": 1,
            "database_pool_max_size": 5,
            "database_connect_timeout_seconds": 10,
        }
    )

    store = create_vector_store(settings)

    assert isinstance(
        store,
        PgVectorStore,
    )
    assert store.dimension == 1024
    assert store.tenant_id == "local"

    token = set_current_principal(
        AuthenticatedPrincipal(
            tenant_id="tenant-a",
            user_id="user-a",
        )
    )

    try:
        assert store.tenant_id == "tenant-a"
    finally:
        reset_current_principal(token)

    assert store.tenant_id == "local"
