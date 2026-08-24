from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.domain.documents import (
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    ImportResult,
    ImportStatus,
    LoadedDocument,
)


def make_document() -> LoadedDocument:
    return LoadedDocument(
        document_id="notes/langgraph.md",
        content_hash="a" * 64,
        metadata=DocumentMetadata(
            source_path="notes/langgraph.md",
            file_name="langgraph.md",
            document_type=DocumentType.MARKDOWN,
            file_size=1024,
            modified_at=datetime.now(UTC),
            title="LangGraph",
        ),
        parts=[
            DocumentPart(
                part_index=0,
                text="LangGraph supports stateful workflows.",
                section_path=["LangGraph"],
            ),
            DocumentPart(
                part_index=1,
                text="Memory can persist between runs.",
                section_path=["LangGraph", "Memory"],
            ),
        ],
    )


def test_loaded_document_combines_part_text() -> None:
    document = make_document()

    assert document.document_id == "notes/langgraph.md"
    assert document.metadata.document_type == DocumentType.MARKDOWN
    assert document.text == (
        "LangGraph supports stateful workflows.\n\nMemory can persist between runs."
    )


def test_sha256_hash_must_have_valid_format() -> None:
    with pytest.raises(ValidationError):
        LoadedDocument(
            document_id="notes/invalid.md",
            content_hash="not-a-valid-hash",
            metadata=DocumentMetadata(
                source_path="notes/invalid.md",
                file_name="invalid.md",
                document_type=DocumentType.MARKDOWN,
                file_size=10,
                modified_at=datetime.now(UTC),
            ),
            parts=[
                DocumentPart(
                    part_index=0,
                    text="Invalid hash example.",
                )
            ],
        )


def test_pdf_page_number_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        DocumentPart(
            part_index=0,
            text="PDF page content.",
            page_number=0,
        )


def test_import_result_records_previous_version() -> None:
    document = make_document()

    result = ImportResult(
        status=ImportStatus.UPDATED,
        document=document,
        previous_content_hash="b" * 64,
    )

    assert result.status == ImportStatus.UPDATED
    assert result.previous_content_hash == "b" * 64
