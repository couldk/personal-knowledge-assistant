from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.chunking import (
    DocumentChunker,
    EmptyChunkingResultError,
    SentenceChunker,
)
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    LoadedDocument,
)


def _make_document(
    *,
    document_type: DocumentType = DocumentType.TEXT,
    content_hash: str = "a" * 64,
    parts: list[DocumentPart] | None = None,
) -> LoadedDocument:
    actual_parts = parts or [
        DocumentPart(
            part_index=0,
            text="RAG combines retrieval with generation.",
        )
    ]

    return LoadedDocument(
        document_id="file:///knowledge/document.txt",
        content_hash=content_hash,
        metadata=DocumentMetadata(
            source_path="knowledge/document.txt",
            file_name="document.txt",
            document_type=document_type,
            file_size=1024,
            modified_at=datetime(
                2026,
                8,
                10,
                tzinfo=UTC,
            ),
            title="Knowledge Document",
        ),
        parts=actual_parts,
    )


def test_sentence_chunker_creates_chunk_with_metadata() -> None:
    document = _make_document()

    chunks = SentenceChunker().split(document)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.text == ("RAG combines retrieval with generation.")
    assert chunk.metadata.content_hash == document.content_hash
    assert chunk.metadata.source_path == (document.metadata.source_path)
    assert chunk.metadata.part_index == 0
    assert chunk.metadata.chunk_index == 0


def test_sentence_chunker_does_not_cross_document_parts() -> None:
    document = _make_document(
        document_type=DocumentType.MARKDOWN,
        parts=[
            DocumentPart(
                part_index=0,
                text="Retrieval finds relevant evidence.",
                section_path=["RAG", "Retrieval"],
            ),
            DocumentPart(
                part_index=1,
                text="Generation produces the final answer.",
                section_path=["RAG", "Generation"],
            ),
        ],
    )

    chunks = SentenceChunker().split(document)

    assert len(chunks) == 2
    assert chunks[0].metadata.part_index == 0
    assert chunks[0].metadata.section_path == [
        "RAG",
        "Retrieval",
    ]
    assert chunks[1].metadata.part_index == 1
    assert chunks[1].metadata.section_path == [
        "RAG",
        "Generation",
    ]

    assert "Generation" not in chunks[0].text
    assert "Retrieval" not in chunks[1].text


def test_sentence_chunker_preserves_pdf_page_numbers() -> None:
    document = _make_document(
        document_type=DocumentType.PDF,
        parts=[
            DocumentPart(
                part_index=0,
                text="First page content.",
                page_number=1,
            ),
            DocumentPart(
                part_index=1,
                text="Second page content.",
                page_number=2,
            ),
        ],
    )

    chunks = SentenceChunker().split(document)

    assert [chunk.metadata.page_number for chunk in chunks] == [
        1,
        2,
    ]


def test_sentence_chunker_splits_long_part() -> None:
    long_text = " ".join(f"Sentence {index} explains retrieval behavior." for index in range(100))
    document = _make_document(
        parts=[
            DocumentPart(
                part_index=0,
                text=long_text,
            )
        ]
    )
    chunker = SentenceChunker(
        config=ChunkingConfig(
            chunk_size=64,
            chunk_overlap=8,
        )
    )

    chunks = chunker.split(document)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert [chunk.metadata.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_sentence_chunker_is_deterministic() -> None:
    document = _make_document()

    chunker = SentenceChunker()

    first_result = chunker.split(document)
    second_result = chunker.split(document)

    assert first_result == second_result


def test_document_version_changes_chunk_ids() -> None:
    first_document = _make_document(
        content_hash="a" * 64,
    )
    second_document = _make_document(
        content_hash="b" * 64,
    )

    chunker = SentenceChunker()

    first_chunks = chunker.split(first_document)
    second_chunks = chunker.split(second_document)

    assert first_chunks[0].text == second_chunks[0].text
    assert first_chunks[0].chunk_id != (second_chunks[0].chunk_id)


def test_sentence_chunker_rejects_blank_document() -> None:
    document = _make_document(
        parts=[
            DocumentPart(
                part_index=0,
                text="   ",
            )
        ]
    )

    with pytest.raises(EmptyChunkingResultError):
        SentenceChunker().split(document)


def test_sentence_chunker_implements_protocol() -> None:
    chunker = SentenceChunker()

    assert isinstance(chunker, DocumentChunker)
