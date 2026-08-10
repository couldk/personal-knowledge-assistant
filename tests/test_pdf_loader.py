from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

import personal_knowledge_assistant.ingestion.pdf_loader as pdf_loader_module
from personal_knowledge_assistant.domain import DocumentType
from personal_knowledge_assistant.ingestion import (
    DocumentParseError,
    EmptyDocumentError,
    LoaderRegistry,
    PdfLoader,
    UnsupportedDocumentTypeError,
)


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakeMetadata:
    def __init__(
        self,
        title: str | None = None,
        author: str | None = None,
    ) -> None:
        self.title = title
        self.author = author


def _install_fake_reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    page_texts: list[str | None],
    title: str | None = None,
    author: str | None = None,
    encrypted: bool = False,
) -> None:
    class FakePdfReader:
        def __init__(
            self,
            stream: BytesIO,
            strict: bool = False,
        ) -> None:
            del stream, strict
            self.pages = [_FakePage(text) for text in page_texts]
            self.metadata = _FakeMetadata(
                title=title,
                author=author,
            )
            self.is_encrypted = encrypted

    monkeypatch.setattr(
        pdf_loader_module,
        "PdfReader",
        FakePdfReader,
    )


def test_pdf_loader_extracts_pages_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reader(
        monkeypatch,
        page_texts=[
            "First page\r\ncontent.",
            None,
            "Third page content.",
        ],
        title="Knowledge Base",
        author="Agent Developer",
    )

    source = tmp_path / "knowledge.pdf"
    source.write_bytes(b"%PDF-1.7 fake test content")

    document = PdfLoader().load(source)

    assert document.metadata.document_type == DocumentType.PDF
    assert document.metadata.title == "Knowledge Base"
    assert document.metadata.extra["author"] == "Agent Developer"
    assert document.metadata.extra["page_count"] == 3
    assert document.metadata.extra["pages_with_text"] == 2

    assert len(document.parts) == 2
    assert document.parts[0].part_index == 0
    assert document.parts[0].page_number == 1
    assert document.parts[0].text == "First page\ncontent."

    assert document.parts[1].part_index == 1
    assert document.parts[1].page_number == 3
    assert document.parts[1].text == "Third page content."

    assert document.content_hash == sha256(source.read_bytes()).hexdigest()


def test_pdf_loader_uses_file_name_as_title(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reader(
        monkeypatch,
        page_texts=["Readable content."],
        title="   ",
    )

    source = tmp_path / "agent-notes.pdf"
    source.write_bytes(b"%PDF-1.7 fake")

    document = PdfLoader().load(source)

    assert document.metadata.title == "agent-notes"


def test_registry_loads_uppercase_pdf_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reader(
        monkeypatch,
        page_texts=["PDF content."],
    )

    source = tmp_path / "NOTES.PDF"
    source.write_bytes(b"%PDF-1.7 fake")

    registry = LoaderRegistry([PdfLoader()])
    document = registry.load(source)

    assert document.metadata.document_type == DocumentType.PDF


def test_pdf_loader_rejects_document_without_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reader(
        monkeypatch,
        page_texts=[None, "", "   \n"],
    )

    source = tmp_path / "scanned.pdf"
    source.write_bytes(b"%PDF-1.7 fake")

    with pytest.raises(EmptyDocumentError):
        PdfLoader().load(source)


def test_pdf_loader_rejects_encrypted_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reader(
        monkeypatch,
        page_texts=["Secret content."],
        encrypted=True,
    )

    source = tmp_path / "secret.pdf"
    source.write_bytes(b"%PDF-1.7 encrypted fake")

    with pytest.raises(
        DocumentParseError,
        match="Encrypted PDF",
    ):
        PdfLoader().load(source)


def test_pdf_loader_rejects_malformed_pdf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"This is not a PDF file.")

    with pytest.raises(DocumentParseError):
        PdfLoader().load(source)


def test_pdf_loader_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Not a PDF.", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        PdfLoader().load(source)
