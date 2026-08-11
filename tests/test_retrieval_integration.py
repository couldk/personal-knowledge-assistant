from collections.abc import Sequence
from pathlib import Path

import pytest

from personal_knowledge_assistant.chunking import SentenceChunker
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    RetrievalQuery,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
)
from personal_knowledge_assistant.ingestion import (
    create_default_ingestion_service,
)
from personal_knowledge_assistant.retrieval import (
    InMemoryRetrievalTracer,
    RetrievalService,
    RetrievalTraceStatus,
)
from personal_knowledge_assistant.vector_store import (
    InMemoryVectorStore,
)


class KeywordEmbeddingProvider:
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
            return [1.0, 0.1, 0.1]

        if "memory" in lowered or "记忆" in lowered:
            return [0.1, 1.0, 0.1]

        return [0.1, 0.1, 1.0]


async def _build_retrieval_system(
    tmp_path: Path,
) -> tuple[
    RetrievalService,
    InMemoryRetrievalTracer,
]:
    vector_source = tmp_path / "vector.txt"
    memory_source = tmp_path / "memory.txt"
    vector_source.write_text(
        "Vector databases support semantic retrieval.",
        encoding="utf-8",
    )
    memory_source.write_text(
        "Long-term memory preserves user preferences.",
        encoding="utf-8",
    )

    ingestion_service = create_default_ingestion_service()
    embedding_provider = KeywordEmbeddingProvider()
    vector_store = InMemoryVectorStore(dimension=3)
    indexing_service = DocumentIndexingService(
        chunker=SentenceChunker(
            ChunkingConfig(
                chunk_size=64,
                chunk_overlap=8,
            )
        ),
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )
    tracer = InMemoryRetrievalTracer()
    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        tracer=tracer,
        embedding_model="test-keyword-model",
    )

    await indexing_service.index(ingestion_service.import_document(vector_source))
    await indexing_service.index(ingestion_service.import_document(memory_source))

    return retrieval_service, tracer


@pytest.mark.asyncio
async def test_indexed_documents_can_be_retrieved_by_query(
    tmp_path: Path,
) -> None:
    retrieval_service, tracer = await _build_retrieval_system(tmp_path)

    results = await retrieval_service.retrieve(
        RetrievalQuery(
            query="如何进行向量检索？",
            top_k=1,
        )
    )

    assert len(results) == 1
    assert results[0].chunk.metadata.file_name == "vector.txt"
    assert len(tracer.traces) == 1
    assert tracer.traces[0].status == (RetrievalTraceStatus.SUCCEEDED)
    assert tracer.traces[0].hits[0].chunk_id == (results[0].chunk.chunk_id)


@pytest.mark.asyncio
async def test_retrieval_applies_metadata_filter(
    tmp_path: Path,
) -> None:
    retrieval_service, _ = await _build_retrieval_system(tmp_path)

    results = await retrieval_service.retrieve(
        RetrievalQuery(
            query="向量检索",
            top_k=5,
            filters={
                "file_name": "memory.txt",
            },
        )
    )

    assert len(results) == 1
    assert results[0].chunk.metadata.file_name == "memory.txt"


@pytest.mark.asyncio
async def test_retrieval_returns_empty_for_unmatched_filter(
    tmp_path: Path,
) -> None:
    retrieval_service, tracer = await _build_retrieval_system(tmp_path)

    results = await retrieval_service.retrieve(
        RetrievalQuery(
            query="向量检索",
            top_k=5,
            filters={
                "file_name": "missing.txt",
            },
        )
    )

    assert results == []
    assert tracer.traces[0].result_count == 0
    assert tracer.traces[0].hits == []
