from pathlib import Path

from personal_knowledge_assistant.domain import ImportStatus
from personal_knowledge_assistant.ingestion import (
    DocumentIngestionService,
    InMemoryDocumentCatalog,
    LoaderRegistry,
    TextLoader,
)


def _build_service() -> tuple[
    DocumentIngestionService,
    InMemoryDocumentCatalog,
]:
    catalog = InMemoryDocumentCatalog()
    registry = LoaderRegistry([TextLoader()])
    service = DocumentIngestionService(
        registry=registry,
        catalog=catalog,
    )

    return service, catalog


def test_first_import_is_created(
    tmp_path: Path,
) -> None:
    service, catalog = _build_service()
    source = tmp_path / "notes.txt"
    source.write_text(
        "First version.",
        encoding="utf-8",
    )

    result = service.import_document(source)

    assert result.status == ImportStatus.CREATED
    assert result.previous_content_hash is None
    assert len(catalog) == 1

    stored_document = catalog.get(result.document.document_id)

    assert stored_document is not None
    assert stored_document == result.document


def test_repeated_import_is_unchanged(
    tmp_path: Path,
) -> None:
    service, catalog = _build_service()
    source = tmp_path / "notes.txt"
    source.write_text(
        "The same content.",
        encoding="utf-8",
    )

    first_result = service.import_document(source)
    second_result = service.import_document(source)

    assert first_result.status == ImportStatus.CREATED
    assert second_result.status == ImportStatus.UNCHANGED
    assert second_result.previous_content_hash == (first_result.document.content_hash)
    assert second_result.document == first_result.document
    assert len(catalog) == 1


def test_changed_document_is_updated(
    tmp_path: Path,
) -> None:
    service, catalog = _build_service()
    source = tmp_path / "notes.txt"
    source.write_text(
        "First version.",
        encoding="utf-8",
    )

    first_result = service.import_document(source)
    first_hash = first_result.document.content_hash

    source.write_text(
        "Second version.",
        encoding="utf-8",
    )

    second_result = service.import_document(source)

    assert second_result.status == ImportStatus.UPDATED
    assert second_result.previous_content_hash == first_hash
    assert second_result.document.content_hash != first_hash
    assert second_result.document.text == "Second version."
    assert len(catalog) == 1

    stored_document = catalog.get(
        second_result.document.document_id,
    )

    assert stored_document == second_result.document


def test_same_content_at_different_paths_is_created_twice(
    tmp_path: Path,
) -> None:
    service, catalog = _build_service()
    first_source = tmp_path / "first.txt"
    second_source = tmp_path / "second.txt"

    first_source.write_text(
        "Shared content.",
        encoding="utf-8",
    )
    second_source.write_text(
        "Shared content.",
        encoding="utf-8",
    )

    first_result = service.import_document(first_source)
    second_result = service.import_document(second_source)

    assert first_result.status == ImportStatus.CREATED
    assert second_result.status == ImportStatus.CREATED
    assert first_result.document.content_hash == (second_result.document.content_hash)
    assert first_result.document.document_id != (second_result.document.document_id)
    assert len(catalog) == 2


def test_catalog_returns_defensive_copy(
    tmp_path: Path,
) -> None:
    service, catalog = _build_service()
    source = tmp_path / "notes.txt"
    source.write_text(
        "Original content.",
        encoding="utf-8",
    )

    result = service.import_document(source)
    fetched_document = catalog.get(
        result.document.document_id,
    )

    assert fetched_document is not None

    fetched_document.parts[0].text = "Changed outside catalog."

    fetched_again = catalog.get(
        result.document.document_id,
    )

    assert fetched_again is not None
    assert fetched_again.parts[0].text == "Original content."
