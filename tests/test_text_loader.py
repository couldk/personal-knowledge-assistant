from hashlib import sha256
from pathlib import Path

import pytest

from personal_knowledge_assistant.domain import DocumentType
from personal_knowledge_assistant.ingestion import (
    DocumentReadError,
    EmptyDocumentError,
    LoaderRegistry,
    TextLoader,
    UnsupportedDocumentTypeError,
)


def test_text_loader_reads_utf8_text(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text(
        "个人知识助手\nKnowledge assistant",
        encoding="utf-8",
    )

    document = TextLoader().load(source)

    assert document.text == "个人知识助手\nKnowledge assistant"
    assert document.metadata.document_type == DocumentType.TEXT
    assert document.metadata.file_name == "notes.txt"
    assert document.metadata.title == "notes"
    assert document.metadata.file_size == source.stat().st_size
    assert document.metadata.extra["encoding"] == "utf-8"
    assert document.document_id == source.resolve().as_uri()
    assert document.parts[0].part_index == 0
    assert document.parts[0].page_number is None
    assert document.parts[0].section_path == []


def test_text_loader_calculates_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "hash.txt"
    source.write_text("hash content", encoding="utf-8")

    document = TextLoader().load(source)

    expected_hash = sha256(source.read_bytes()).hexdigest()

    assert document.content_hash == expected_hash


def test_text_loader_removes_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "bom.txt"
    source.write_bytes(b"\xef\xbb\xbfKnowledge assistant")

    document = TextLoader().load(source)

    assert document.text == "Knowledge assistant"
    assert not document.text.startswith("\ufeff")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"   ",
        b"\n\n",
        b"\t",
    ],
)
def test_text_loader_rejects_empty_content(
    tmp_path: Path,
    content: bytes,
) -> None:
    source = tmp_path / "empty.txt"
    source.write_bytes(content)

    with pytest.raises(EmptyDocumentError):
        TextLoader().load(source)


def test_text_loader_rejects_invalid_utf8(tmp_path: Path) -> None:
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe\x00\x00")

    with pytest.raises(DocumentReadError):
        TextLoader().load(source)


def test_text_loader_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.md"
    source.write_text("Markdown content", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        TextLoader().load(source)


def test_registry_loads_text_document(tmp_path: Path) -> None:
    source = tmp_path / "registry.txt"
    source.write_text(
        "Loaded through registry",
        encoding="utf-8",
    )

    registry = LoaderRegistry([TextLoader()])
    document = registry.load(source)

    assert document.text == "Loaded through registry"
    assert document.metadata.document_type == DocumentType.TEXT
