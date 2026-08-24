from hashlib import sha256
from pathlib import Path

import pytest

from personal_knowledge_assistant.domain import DocumentType
from personal_knowledge_assistant.ingestion import (
    EmptyDocumentError,
    LoaderRegistry,
    MarkdownLoader,
    UnsupportedDocumentTypeError,
)


def test_markdown_loader_builds_heading_hierarchy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "langgraph.md"
    source.write_text(
        (
            "# LangGraph\n\n"
            "LangGraph is a workflow framework.\n\n"
            "## Memory\n\n"
            "Memory stores state.\n\n"
            "### Long-term Memory\n\n"
            "Long-term memory persists across sessions.\n\n"
            "## Tools\n\n"
            "Tools connect external systems.\n"
        ),
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert document.metadata.title == "LangGraph"
    assert document.metadata.document_type == DocumentType.MARKDOWN
    assert len(document.parts) == 4

    assert document.parts[0].section_path == ["LangGraph"]
    assert document.parts[1].section_path == [
        "LangGraph",
        "Memory",
    ]
    assert document.parts[2].section_path == [
        "LangGraph",
        "Memory",
        "Long-term Memory",
    ]
    assert document.parts[3].section_path == [
        "LangGraph",
        "Tools",
    ]


def test_markdown_loader_preserves_preamble(
    tmp_path: Path,
) -> None:
    source = tmp_path / "preamble.md"
    source.write_text(
        "Document introduction.\n\n# Main Title\n\nBody.",
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert len(document.parts) == 2
    assert document.parts[0].text == "Document introduction."
    assert document.parts[0].section_path == []
    assert document.parts[1].section_path == ["Main Title"]


def test_markdown_loader_uses_file_name_without_h1(
    tmp_path: Path,
) -> None:
    source = tmp_path / "architecture.md"
    source.write_text(
        "## Components\n\nComponent details.",
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert document.metadata.title == "architecture"
    assert document.parts[0].section_path == ["Components"]


def test_markdown_loader_handles_document_without_headings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain.md"
    source.write_text(
        "A Markdown document without headings.",
        encoding="utf-8",
    )

    document = MarkdownLoader().load(source)

    assert len(document.parts) == 1
    assert document.parts[0].section_path == []
    assert document.parts[0].text == ("A Markdown document without headings.")


def test_markdown_loader_ignores_heading_inside_code_fence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "code.md"
    source.write_text(
        ("# Real Heading\n\n```python\n# Not a Markdown heading\nprint('hello')\n```\n"),
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert len(document.parts) == 1
    assert document.parts[0].section_path == ["Real Heading"]
    assert "# Not a Markdown heading" in document.parts[0].text


def test_markdown_loader_supports_setext_heading(
    tmp_path: Path,
) -> None:
    source = tmp_path / "setext.md"
    source.write_text(
        "Setext Title\n============\n\nBody.",
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert document.metadata.title == "Setext Title"
    assert document.parts[0].section_path == ["Setext Title"]
    assert "Body." in document.parts[0].text


def test_markdown_loader_calculates_raw_content_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hash.md"
    source.write_text(
        "# Hash\n\nContent.",
        encoding="utf-8",
        newline="\n",
    )

    document = MarkdownLoader().load(source)

    assert document.content_hash == sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "file_name",
    [
        "notes.md",
        "notes.markdown",
        "NOTES.MD",
    ],
)
def test_registry_loads_supported_markdown_extensions(
    tmp_path: Path,
    file_name: str,
) -> None:
    source = tmp_path / file_name
    source.write_text(
        "# Notes\n\nContent.",
        encoding="utf-8",
    )

    registry = LoaderRegistry([MarkdownLoader()])
    document = registry.load(source)

    assert document.metadata.document_type == (DocumentType.MARKDOWN)


def test_markdown_loader_rejects_empty_document(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty.md"
    source.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        MarkdownLoader().load(source)


def test_markdown_loader_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("# Notes", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        MarkdownLoader().load(source)
