from pathlib import Path

import pytest

from personal_knowledge_assistant.domain import (
    DocumentType,
    ImportStatus,
)
from personal_knowledge_assistant.ingestion import (
    DocumentLoader,
    InMemoryDocumentCatalog,
    MarkdownLoader,
    PdfLoader,
    TextLoader,
    UnsupportedDocumentTypeError,
    create_default_ingestion_service,
    create_default_loader_registry,
)


@pytest.mark.parametrize(
    ("file_name", "expected_loader_type"),
    [
        ("notes.txt", TextLoader),
        ("NOTES.TXT", TextLoader),
        ("notes.md", MarkdownLoader),
        ("notes.markdown", MarkdownLoader),
        ("document.pdf", PdfLoader),
    ],
)
def test_default_registry_contains_builtin_loaders(
    file_name: str,
    expected_loader_type: type[DocumentLoader],
) -> None:
    registry = create_default_loader_registry()

    loader = registry.get_loader(Path(file_name))

    assert isinstance(loader, expected_loader_type)


def test_default_service_imports_text_and_markdown(
    tmp_path: Path,
) -> None:
    service = create_default_ingestion_service()

    text_source = tmp_path / "notes.txt"
    markdown_source = tmp_path / "guide.md"

    text_source.write_text(
        "Personal knowledge assistant.",
        encoding="utf-8",
    )
    markdown_source.write_text(
        "# Agent Guide\n\n## Memory\n\nMemory preserves useful information.",
        encoding="utf-8",
        newline="\n",
    )

    text_result = service.import_document(text_source)
    markdown_result = service.import_document(markdown_source)
    repeated_result = service.import_document(text_source)

    assert text_result.status == ImportStatus.CREATED
    assert text_result.document.metadata.document_type == DocumentType.TEXT

    assert markdown_result.status == ImportStatus.CREATED
    assert markdown_result.document.metadata.document_type == DocumentType.MARKDOWN
    assert markdown_result.document.metadata.title == "Agent Guide"
    assert markdown_result.document.parts[1].section_path == [
        "Agent Guide",
        "Memory",
    ]

    assert repeated_result.status == ImportStatus.UNCHANGED


def test_default_service_accepts_custom_catalog(
    tmp_path: Path,
) -> None:
    catalog = InMemoryDocumentCatalog()
    service = create_default_ingestion_service(
        catalog=catalog,
    )

    source = tmp_path / "notes.txt"
    source.write_text(
        "Catalog integration.",
        encoding="utf-8",
    )

    result = service.import_document(source)

    assert result.status == ImportStatus.CREATED
    assert len(catalog) == 1

    stored_document = catalog.get(
        result.document.document_id,
    )

    assert stored_document == result.document


def test_default_service_rejects_unsupported_document(
    tmp_path: Path,
) -> None:
    service = create_default_ingestion_service()
    source = tmp_path / "notes.docx"
    source.write_bytes(b"Unsupported document.")

    with pytest.raises(UnsupportedDocumentTypeError):
        service.import_document(source)
