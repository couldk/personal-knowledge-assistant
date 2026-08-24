from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

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
    DocumentParseError,
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)


class PdfLoader(DocumentLoader):
    """读取PDF文件并保留每一页的页码信息。"""

    supported_extensions = frozenset({".pdf"})

    def load(self, path: Path) -> LoadedDocument:
        """读取PDF并转换为统一文档对象。"""

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
            reader = PdfReader(
                BytesIO(raw_content),
                strict=False,
            )
        except (PyPdfError, ValueError, TypeError) as exc:
            raise DocumentParseError(
                validated_path,
                str(exc),
            ) from exc

        if reader.is_encrypted:
            raise DocumentParseError(
                validated_path,
                "Encrypted PDF documents are not supported.",
            )

        try:
            page_count = len(reader.pages)
            parts: list[DocumentPart] = []

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                extracted_text = page.extract_text()
                page_text = self._normalize_text(extracted_text or "")

                # 空白页不产生DocumentPart，但后续页仍保留原始页码。
                if not page_text:
                    continue

                parts.append(
                    DocumentPart(
                        part_index=len(parts),
                        text=page_text,
                        page_number=page_number,
                    )
                )

            pdf_metadata = reader.metadata
            raw_title = pdf_metadata.title if pdf_metadata is not None else None
            raw_author = pdf_metadata.author if pdf_metadata is not None else None
        except (PyPdfError, ValueError, TypeError) as exc:
            raise DocumentParseError(
                validated_path,
                str(exc),
            ) from exc

        if not parts:
            raise EmptyDocumentError(validated_path)

        title = (
            raw_title.strip()
            if raw_title is not None and raw_title.strip()
            else validated_path.stem
        )

        extra: dict[str, str | int | float | bool | None] = {
            "parser": "pypdf",
            "page_count": page_count,
            "pages_with_text": len(parts),
            "encrypted": False,
        }

        if raw_author is not None and raw_author.strip():
            extra["author"] = raw_author.strip()

        return LoadedDocument(
            document_id=validated_path.as_uri(),
            content_hash=sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                source_path=validated_path.as_posix(),
                file_name=validated_path.name,
                document_type=DocumentType.PDF,
                file_size=stat.st_size,
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=UTC,
                ),
                title=title,
                extra=extra,
            ),
            parts=parts,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一PDF提取结果中的换行符。"""

        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
