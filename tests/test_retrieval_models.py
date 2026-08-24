from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    ChunkMetadata,
    DocumentChunk,
    DocumentType,
    RetrievalQuery,
    SearchResult,
    create_chunk_id,
)

CONTENT_HASH = "a" * 64


def _create_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        content_hash=CONTENT_HASH,
        source_path="docs/rag.md",
        file_name="rag.md",
        document_type=DocumentType.MARKDOWN,
        modified_at=datetime(
            2026,
            8,
            10,
            tzinfo=UTC,
        ),
        part_index=1,
        chunk_index=2,
        section_path=["RAG", "Retrieval"],
    )


def test_create_chunk_id_is_deterministic() -> None:
    first = create_chunk_id(
        document_id="document-001",
        content_hash=CONTENT_HASH,
        part_index=1,
        chunk_index=2,
    )
    second = create_chunk_id(
        document_id="document-001",
        content_hash=CONTENT_HASH,
        part_index=1,
        chunk_index=2,
    )

    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("content_hash", "b" * 64),
        ("part_index", 2),
        ("chunk_index", 3),
    ],
)
def test_chunk_id_changes_when_identity_changes(
    changed_field: str,
    changed_value: str | int,
) -> None:
    original_values: dict[str, str | int] = {
        "document_id": "document-001",
        "content_hash": CONTENT_HASH,
        "part_index": 1,
        "chunk_index": 2,
    }
    changed_values = {
        **original_values,
        changed_field: changed_value,
    }

    original = create_chunk_id(
        document_id=str(original_values["document_id"]),
        content_hash=str(original_values["content_hash"]),
        part_index=int(original_values["part_index"]),
        chunk_index=int(original_values["chunk_index"]),
    )
    changed = create_chunk_id(
        document_id=str(changed_values["document_id"]),
        content_hash=str(changed_values["content_hash"]),
        part_index=int(changed_values["part_index"]),
        chunk_index=int(changed_values["chunk_index"]),
    )

    assert changed != original


def test_document_chunk_preserves_source_metadata() -> None:
    chunk_id = create_chunk_id(
        document_id="document-001",
        content_hash=CONTENT_HASH,
        part_index=1,
        chunk_index=2,
    )

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        text="RAG combines retrieval and generation.",
        metadata=_create_metadata(),
    )

    assert chunk.metadata.file_name == "rag.md"
    assert chunk.metadata.section_path == [
        "RAG",
        "Retrieval",
    ]


def test_document_chunk_rejects_blank_text() -> None:
    chunk_id = create_chunk_id(
        document_id="document-001",
        content_hash=CONTENT_HASH,
        part_index=1,
        chunk_index=2,
    )

    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id=chunk_id,
            document_id="document-001",
            text="   ",
            metadata=_create_metadata(),
        )


def test_chunking_config_has_safe_defaults() -> None:
    config = ChunkingConfig()

    assert config.chunk_size == 512
    assert config.chunk_overlap == 64


def test_chunking_config_rejects_overlap_equal_to_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(
            chunk_size=128,
            chunk_overlap=128,
        )


@pytest.mark.parametrize("top_k", [0, 21])
def test_retrieval_query_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(
            query="What is RAG?",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "score",
    [
        -1.01,
        1.01,
        float("nan"),
    ],
)
def test_search_result_rejects_invalid_score(
    score: float,
) -> None:
    chunk_id = create_chunk_id(
        document_id="document-001",
        content_hash=CONTENT_HASH,
        part_index=1,
        chunk_index=2,
    )
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id="document-001",
        text="RAG content.",
        metadata=_create_metadata(),
    )

    with pytest.raises(ValidationError):
        SearchResult(
            chunk=chunk,
            score=score,
        )
