from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_knowledge_assistant.application import (
    DocumentImportService,
    InvalidUploadFileNameError,
    UnsupportedUploadTypeError,
    UploadTooLargeError,
)
from personal_knowledge_assistant.chunking import (
    SentenceChunker,
)
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    ImportStatus,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
    IndexingStatus,
)
from personal_knowledge_assistant.ingestion import (
    create_default_ingestion_service,
)
from personal_knowledge_assistant.vector_store import (
    InMemoryVectorStore,
)


class DeterministicEmbeddingProvider:
    """测试用确定性Embedding。"""

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
    def _embed(
        text: str,
    ) -> list[float]:
        lowered = text.casefold()

        if "vector" in lowered:
            return [1.0, 0.0, 0.0]

        if "memory" in lowered:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def create_service(
    upload_directory: Path,
) -> tuple[
    DocumentImportService,
    InMemoryVectorStore,
]:
    vector_store = InMemoryVectorStore(
        dimension=3,
    )

    indexing_service = DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=64,
                chunk_overlap=8,
            )
        ),
        embedding_provider=(DeterministicEmbeddingProvider()),
        vector_store=vector_store,
    )

    service = DocumentImportService(
        ingestion_service=(create_default_ingestion_service()),
        indexing_service=indexing_service,
        upload_directory=upload_directory,
        max_upload_bytes=1024,
    )

    return service, vector_store


@pytest.mark.asyncio
async def test_upload_is_saved_and_indexed(
    tmp_path: Path,
) -> None:
    service, vector_store = create_service(tmp_path)

    outcome = await service.import_upload(
        file_name="knowledge.txt",
        content=(b"Vector databases provide semantic search."),
    )

    results = await vector_store.search(
        [1.0, 0.0, 0.0],
        limit=5,
    )

    assert outcome.import_status is (ImportStatus.CREATED)
    assert outcome.indexing_status is (IndexingStatus.INDEXED)
    assert outcome.chunk_count > 0
    assert (tmp_path / "knowledge.txt").is_file()
    assert results
    assert results[0].chunk.document_id == outcome.document_id


@pytest.mark.asyncio
async def test_repeated_upload_is_skipped(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)
    content = b"Memory can persist between runs."

    first = await service.import_upload(
        file_name="memory.txt",
        content=content,
    )
    second = await service.import_upload(
        file_name="memory.txt",
        content=content,
    )

    assert first.import_status is (ImportStatus.CREATED)
    assert second.import_status is (ImportStatus.UNCHANGED)
    assert second.indexing_status is (IndexingStatus.SKIPPED)
    assert second.chunk_count == 0


@pytest.mark.asyncio
async def test_force_reindex_indexes_unchanged_upload(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)
    content = b"Memory can be reindexed."

    await service.import_upload(
        file_name="memory.txt",
        content=content,
    )
    forced = await service.import_upload(
        file_name="memory.txt",
        content=content,
        force_reindex=True,
    )

    assert forced.import_status is ImportStatus.UNCHANGED
    assert forced.indexing_status is IndexingStatus.INDEXED
    assert forced.chunk_count > 0


@pytest.mark.asyncio
async def test_updated_upload_deactivates_old_chunks(
    tmp_path: Path,
) -> None:
    service, vector_store = create_service(tmp_path)

    first = await service.import_upload(
        file_name="knowledge.txt",
        content=b"Vector databases.",
    )
    second = await service.import_upload(
        file_name="knowledge.txt",
        content=b"Memory persists.",
    )

    old_results = await vector_store.search(
        [1.0, 0.0, 0.0],
        limit=5,
        filters={
            "content_hash": first.content_hash,
        },
    )
    new_results = await vector_store.search(
        [0.0, 1.0, 0.0],
        limit=5,
        filters={
            "content_hash": second.content_hash,
        },
    )

    assert second.import_status is (ImportStatus.UPDATED)
    assert second.deactivated_chunk_count > 0
    assert old_results == []
    assert new_results


@pytest.mark.asyncio
async def test_upload_rejects_unsafe_name(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)

    with pytest.raises(InvalidUploadFileNameError):
        await service.import_upload(
            file_name="../secret.txt",
            content=b"secret",
        )


@pytest.mark.asyncio
async def test_upload_rejects_windows_reserved_name(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)

    with pytest.raises(InvalidUploadFileNameError):
        await service.import_upload(
            file_name="CON.backup.txt",
            content=b"secret",
        )


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_type(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)

    with pytest.raises(UnsupportedUploadTypeError):
        await service.import_upload(
            file_name="program.exe",
            content=b"binary",
        )


@pytest.mark.asyncio
async def test_upload_rejects_large_file(
    tmp_path: Path,
) -> None:
    service, _ = create_service(tmp_path)

    with pytest.raises(UploadTooLargeError):
        await service.import_upload(
            file_name="large.txt",
            content=b"x" * 1025,
        )
