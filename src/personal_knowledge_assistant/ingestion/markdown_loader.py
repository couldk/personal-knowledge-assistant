from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from personal_knowledge_assistant.domain import (
    DocumentMetadata,
    DocumentPart,
    DocumentType,
    LoadedDocument,
)
from personal_knowledge_assistant.ingestion.base import (
    DocumentLoader,
    validate_source_path,
)
from personal_knowledge_assistant.ingestion.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)


@dataclass(frozen=True, slots=True)
class _Heading:
    """从Markdown Token中提取出的标题信息。"""

    level: int
    title: str
    start_line: int
    end_line: int


class MarkdownLoader(DocumentLoader):
    """读取Markdown并按标题层级拆分内容。"""

    supported_extensions = frozenset({".md", ".markdown"})

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark")

    def load(self, path: Path) -> LoadedDocument:
        """读取Markdown文件并转换为统一文档对象。"""

        validated_path = validate_source_path(path)

        if not self.supports(validated_path):
            raise UnsupportedDocumentTypeError(validated_path)

        try:
            raw_content = validated_path.read_bytes()
            stat = validated_path.stat()
        except OSError as exc:
            raise DocumentReadError(
                validated_path,
                str(exc),
            ) from exc

        try:
            text = raw_content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentReadError(
                validated_path,
                "The document is not valid UTF-8 Markdown.",
            ) from exc

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise EmptyDocumentError(validated_path)

        tokens = self._parser.parse(text)
        headings = self._extract_headings(tokens)
        parts = self._build_parts(text, headings)

        title = next(
            (heading.title for heading in headings if heading.level == 1),
            validated_path.stem,
        )

        return LoadedDocument(
            document_id=validated_path.as_uri(),
            content_hash=sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                source_path=validated_path.as_posix(),
                file_name=validated_path.name,
                document_type=DocumentType.MARKDOWN,
                file_size=stat.st_size,
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=UTC,
                ),
                title=title,
                extra={
                    "encoding": "utf-8",
                    "parser": "markdown-it-py",
                    "part_strategy": "heading-sections",
                },
            ),
            parts=parts,
        )

    @staticmethod
    def _extract_headings(tokens: list[Token]) -> list[_Heading]:
        """从Token Stream中提取标题及源码行号。"""

        headings: list[_Heading] = []

        for index, token in enumerate(tokens):
            if token.type != "heading_open":
                continue

            if token.map is None:
                continue

            if index + 1 >= len(tokens):
                continue

            inline_token = tokens[index + 1]

            if inline_token.type != "inline":
                continue

            title = inline_token.content.strip()

            # 空标题不能形成有意义的section_path。
            if not title:
                continue

            start_line, end_line = token.map
            level = int(token.tag.removeprefix("h"))

            headings.append(
                _Heading(
                    level=level,
                    title=title,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return headings

    @staticmethod
    def _build_parts(
        text: str,
        headings: list[_Heading],
    ) -> list[DocumentPart]:
        """根据标题位置构建文档内容单元。"""

        if not headings:
            return [
                DocumentPart(
                    part_index=0,
                    text=text.strip(),
                )
            ]

        lines = text.splitlines()
        parts: list[DocumentPart] = []

        first_heading = headings[0]
        preamble = "\n".join(lines[: first_heading.start_line]).strip()

        if preamble:
            parts.append(
                DocumentPart(
                    part_index=len(parts),
                    text=preamble,
                )
            )

        heading_stack: list[tuple[int, str]] = []

        for index, heading in enumerate(headings):
            heading_stack = [item for item in heading_stack if item[0] < heading.level]
            heading_stack.append((heading.level, heading.title))

            content_end = (
                headings[index + 1].start_line if index + 1 < len(headings) else len(lines)
            )

            body = "\n".join(lines[heading.end_line : content_end]).strip()

            part_text = heading.title

            if body:
                part_text = f"{heading.title}\n\n{body}"

            parts.append(
                DocumentPart(
                    part_index=len(parts),
                    text=part_text,
                    section_path=[title for _, title in heading_stack],
                )
            )

        return parts
