from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_knowledge_assistant.chunking import SentenceChunker
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
    """根据关键词返回固定三维向量。"""

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
        lowered = text.lower()

        if "vector" in lowered or "向量" in lowered:
            return [1.0, 0.0, 0.0]

        if "memory" in lowered or "记忆" in lowered:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def _create_service(
    vector_store: InMemoryVectorStore,
) -> DocumentIndexingService:
    return DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=64,
                chunk_overlap=8,
            )
        ),
        embedding_provider=(DeterministicEmbeddingProvider()),
        vector_store=vector_store,
    )


@pytest.mark.asyncio
async def test_text_file_can_be_imported_indexed_and_searched(
    tmp_path: Path,
) -> None:
    source = tmp_path / "knowledge.txt"
    source.write_text(
        "Vector databases provide semantic retrieval.",
        encoding="utf-8",
    )
    ingestion_service = create_default_ingestion_service()
    import_result = ingestion_service.import_document(source)
    vector_store = InMemoryVectorStore(
        dimension=3,
    )
    indexing_service = _create_service(vector_store)

    indexing_result = await indexing_service.index(import_result)
    search_results = await vector_store.search(
        [1.0, 0.0, 0.0],
        limit=5,
    )

    assert import_result.status == ImportStatus.CREATED
    assert indexing_result.status == IndexingStatus.INDEXED
    assert indexing_result.chunk_count > 0
    assert search_results
    assert search_results[0].chunk.document_id == import_result.document.document_id
    assert search_results[0].chunk.metadata.file_name == "knowledge.txt"


@pytest.mark.asyncio
async def test_repeated_import_is_skipped_without_new_vectors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "knowledge.txt"
    source.write_text(
        "Memory preserves useful information.",
        encoding="utf-8",
    )
    ingestion_service = create_default_ingestion_service()
    created_result = ingestion_service.import_document(source)
    unchanged_result = ingestion_service.import_document(source)
    vector_store = InMemoryVectorStore(
        dimension=3,
    )
    indexing_service = _create_service(vector_store)

    first_indexing_result = await indexing_service.index(created_result)
    second_indexing_result = await indexing_service.index(unchanged_result)
    search_results = await vector_store.search(
        [0.0, 1.0, 0.0],
        limit=5,
    )

    assert first_indexing_result.status == (IndexingStatus.INDEXED)
    assert second_indexing_result.status == (IndexingStatus.SKIPPED)
    assert second_indexing_result.chunk_count == 0
    assert len(search_results) == (first_indexing_result.chunk_count)


@pytest.mark.asyncio
async def test_updated_file_replaces_active_document_version(
    tmp_path: Path,
) -> None:
    source = tmp_path / "knowledge.txt"
    source.write_text(
        "Vector databases provide semantic retrieval.",
        encoding="utf-8",
    )
    ingestion_service = create_default_ingestion_service()
    created_result = ingestion_service.import_document(source)
    vector_store = InMemoryVectorStore(
        dimension=3,
    )
    indexing_service = _create_service(vector_store)

    await indexing_service.index(created_result)

    source.write_text(
        "Memory preserves useful information.",
        encoding="utf-8",
    )
    updated_result = ingestion_service.import_document(source)
    indexing_result = await indexing_service.index(updated_result)

    old_results = await vector_store.search(
        [1.0, 0.0, 0.0],
        limit=5,
        filters={"content_hash": (created_result.document.content_hash)},
    )
    new_results = await vector_store.search(
        [0.0, 1.0, 0.0],
        limit=5,
        filters={"content_hash": (updated_result.document.content_hash)},
    )

    assert updated_result.status == ImportStatus.UPDATED
    assert indexing_result.status == IndexingStatus.INDEXED
    assert indexing_result.deactivated_chunk_count > 0
    assert old_results == []
    assert new_results
    assert all(
        result.chunk.metadata.content_hash == updated_result.document.content_hash
        for result in new_results
    )
