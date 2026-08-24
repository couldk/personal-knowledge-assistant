from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.domain import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    ImportResult,
    ImportStatus,
    LoadedDocument,
    SearchResult,
    create_chunk_id,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
    EmbeddingCountMismatchError,
    IndexingStatus,
)


class FakeChunker:
    """记录调用的可控切块器。"""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        events: list[str] | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._events = events
        self.documents: list[LoadedDocument] = []

    def split(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        self.documents.append(document)

        if self._events is not None:
            self._events.append("chunk")

        return [chunk.model_copy(deep=True) for chunk in self._chunks]


class FakeEmbeddingProvider:
    """不会访问网络的可控 Embedding Provider。"""

    def __init__(
        self,
        embeddings: Sequence[Sequence[float]],
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self._embeddings = [list(embedding) for embedding in embeddings]
        self._error = error
        self._events = events
        self.text_batches: list[list[str]] = []

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        self.text_batches.append(list(texts))

        if self._events is not None:
            self._events.append("embed")

        if self._error is not None:
            raise self._error

        return [embedding.copy() for embedding in self._embeddings]

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        del query
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    """记录写入和停用操作的向量存储。"""

    def __init__(
        self,
        *,
        deactivated_count: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self._deactivated_count = deactivated_count
        self._events = events
        self.upsert_calls: list[
            tuple[
                list[DocumentChunk],
                list[list[float]],
            ]
        ] = []
        self.deactivate_calls: list[str] = []

    async def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if self._events is not None:
            self._events.append("upsert")

        self.upsert_calls.append(
            (
                [chunk.model_copy(deep=True) for chunk in chunks],
                [list(embedding) for embedding in embeddings],
            )
        )

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        del query_embedding, limit, filters
        return []

    async def deactivate_document(
        self,
        document_id: str,
    ) -> int:
        if self._events is not None:
            self._events.append("deactivate")

        self.deactivate_calls.append(document_id)
        return self._deactivated_count


def _make_document(
    *,
    content_hash: str = "a" * 64,
    text: str = "Personal knowledge assistant.",
) -> LoadedDocument:
    return LoadedDocument(
        document_id="file:///documents/notes.txt",
        content_hash=content_hash,
        metadata=DocumentMetadata(
            source_path="D:/documents/notes.txt",
            file_name="notes.txt",
            document_type=DocumentType.TEXT,
            file_size=len(text.encode("utf-8")),
            modified_at=datetime(
                2026,
                8,
                11,
                10,
                0,
                tzinfo=UTC,
            ),
        ),
        parts=[
            DocumentPart(
                part_index=0,
                text=text,
            )
        ],
    )


def _make_chunk(
    document: LoadedDocument,
    *,
    chunk_index: int = 0,
    text: str = "Personal knowledge assistant.",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=create_chunk_id(
            document_id=document.document_id,
            content_hash=document.content_hash,
            part_index=0,
            chunk_index=chunk_index,
        ),
        document_id=document.document_id,
        text=text,
        metadata=ChunkMetadata(
            content_hash=document.content_hash,
            source_path=document.metadata.source_path,
            file_name=document.metadata.file_name,
            document_type=document.metadata.document_type,
            modified_at=document.metadata.modified_at,
            part_index=0,
            chunk_index=chunk_index,
        ),
    )


def _make_import_result(
    status: ImportStatus,
    *,
    document: LoadedDocument | None = None,
) -> ImportResult:
    selected_document = document or _make_document()

    return ImportResult(
        status=status,
        document=selected_document,
        previous_content_hash=("b" * 64 if status == ImportStatus.UPDATED else None),
    )


@pytest.mark.asyncio
async def test_created_document_is_chunked_embedded_and_stored() -> None:
    document = _make_document()
    chunks = [
        _make_chunk(document, chunk_index=0),
        _make_chunk(
            document,
            chunk_index=1,
            text="Second chunk.",
        ),
    ]
    chunker = FakeChunker(chunks)
    embedding_provider = FakeEmbeddingProvider(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = await service.index(
        _make_import_result(
            ImportStatus.CREATED,
            document=document,
        )
    )

    assert result.status == IndexingStatus.INDEXED
    assert result.document_id == document.document_id
    assert result.content_hash == document.content_hash
    assert result.chunk_count == 2
    assert result.deactivated_chunk_count == 0
    assert chunker.documents == [document]
    assert embedding_provider.text_batches == [
        [
            "Personal knowledge assistant.",
            "Second chunk.",
        ]
    ]
    assert len(vector_store.upsert_calls) == 1
    stored_chunks, stored_embeddings = vector_store.upsert_calls[0]
    assert stored_chunks == chunks
    assert stored_embeddings == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert vector_store.deactivate_calls == []


@pytest.mark.asyncio
async def test_unchanged_document_is_skipped() -> None:
    document = _make_document()
    chunker = FakeChunker([_make_chunk(document)])
    embedding_provider = FakeEmbeddingProvider([[1.0, 0.0, 0.0]])
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = await service.index(
        _make_import_result(
            ImportStatus.UNCHANGED,
            document=document,
        )
    )

    assert result.status == IndexingStatus.SKIPPED
    assert result.chunk_count == 0
    assert result.deactivated_chunk_count == 0
    assert chunker.documents == []
    assert embedding_provider.text_batches == []
    assert vector_store.deactivate_calls == []
    assert vector_store.upsert_calls == []


@pytest.mark.asyncio
async def test_updated_document_deactivates_old_chunks_before_upsert() -> None:
    events: list[str] = []
    document = _make_document(
        content_hash="c" * 64,
        text="Updated content.",
    )
    chunk = _make_chunk(
        document,
        text="Updated content.",
    )
    chunker = FakeChunker(
        [chunk],
        events,
    )
    embedding_provider = FakeEmbeddingProvider(
        [[0.0, 1.0, 0.0]],
        events=events,
    )
    vector_store = FakeVectorStore(
        deactivated_count=3,
        events=events,
    )
    service = DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    result = await service.index(
        _make_import_result(
            ImportStatus.UPDATED,
            document=document,
        )
    )

    assert events == [
        "chunk",
        "embed",
        "deactivate",
        "upsert",
    ]
    assert vector_store.deactivate_calls == [document.document_id]
    assert result.status == IndexingStatus.INDEXED
    assert result.chunk_count == 1
    assert result.deactivated_chunk_count == 3


@pytest.mark.asyncio
async def test_embedding_failure_does_not_deactivate_old_chunks() -> None:
    document = _make_document(
        content_hash="c" * 64,
    )
    chunker = FakeChunker([_make_chunk(document)])
    embedding_provider = FakeEmbeddingProvider(
        [],
        error=RuntimeError("embedding unavailable"),
    )
    vector_store = FakeVectorStore(
        deactivated_count=2,
    )
    service = DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding unavailable",
    ):
        await service.index(
            _make_import_result(
                ImportStatus.UPDATED,
                document=document,
            )
        )

    assert vector_store.deactivate_calls == []
    assert vector_store.upsert_calls == []


@pytest.mark.asyncio
async def test_embedding_count_mismatch_does_not_change_store() -> None:
    document = _make_document(
        content_hash="c" * 64,
    )
    chunks = [
        _make_chunk(document, chunk_index=0),
        _make_chunk(
            document,
            chunk_index=1,
            text="Second chunk.",
        ),
    ]
    chunker = FakeChunker(chunks)
    embedding_provider = FakeEmbeddingProvider([[1.0, 0.0, 0.0]])
    vector_store = FakeVectorStore(
        deactivated_count=2,
    )
    service = DocumentIndexingService(
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    with pytest.raises(
        EmbeddingCountMismatchError,
        match="Expected 2 embeddings, received 1",
    ):
        await service.index(
            _make_import_result(
                ImportStatus.UPDATED,
                document=document,
            )
        )

    assert vector_store.deactivate_calls == []
    assert vector_store.upsert_calls == []
