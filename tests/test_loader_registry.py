from datetime import UTC, datetime
from pathlib import Path

import pytest

from personal_knowledge_assistant.domain import (
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    LoadedDocument,
)
from personal_knowledge_assistant.ingestion import (
    DocumentLoader,
    DocumentNotFoundError,
    DuplicateLoaderRegistrationError,
    LoaderRegistry,
    UnsupportedDocumentTypeError,
)


class FakeTextLoader(DocumentLoader):
    supported_extensions = frozenset({".txt"})

    def load(self, path: Path) -> LoadedDocument:
        stat = path.stat()

        return LoadedDocument(
            document_id=path.name,
            content_hash="a" * 64,
            metadata=DocumentMetadata(
                source_path=path.as_posix(),
                file_name=path.name,
                document_type=DocumentType.TEXT,
                file_size=stat.st_size,
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=UTC,
                ),
                title=path.stem,
            ),
            parts=[
                DocumentPart(
                    part_index=0,
                    text=path.read_text(encoding="utf-8"),
                )
            ],
        )


def test_loader_supports_extension_case_insensitively() -> None:
    loader = FakeTextLoader()

    assert loader.supports(Path("notes.txt"))
    assert loader.supports(Path("NOTES.TXT"))
    assert not loader.supports(Path("notes.md"))


def test_registry_returns_loader_for_extension() -> None:
    loader = FakeTextLoader()
    registry = LoaderRegistry([loader])

    selected_loader = registry.get_loader(Path("notes.TXT"))

    assert selected_loader is loader


def test_registry_rejects_duplicate_extension() -> None:
    registry = LoaderRegistry([FakeTextLoader()])

    with pytest.raises(DuplicateLoaderRegistrationError):
        registry.register(FakeTextLoader())


def test_registry_rejects_unsupported_file_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("name,value\nexample,1", encoding="utf-8")

    registry = LoaderRegistry([FakeTextLoader()])

    with pytest.raises(UnsupportedDocumentTypeError):
        registry.load(source)


def test_registry_rejects_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"
    registry = LoaderRegistry([FakeTextLoader()])

    with pytest.raises(DocumentNotFoundError):
        registry.load(missing_file)


def test_registry_delegates_loading(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Knowledge assistant", encoding="utf-8")

    registry = LoaderRegistry([FakeTextLoader()])
    document = registry.load(source)

    assert document.metadata.file_name == "notes.txt"
    assert document.metadata.document_type == DocumentType.TEXT
    assert document.text == "Knowledge assistant"
