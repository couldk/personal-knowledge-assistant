from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

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


class TextLoader(DocumentLoader):
    """读取UTF-8纯文本文件。"""

    supported_extensions = frozenset({".txt"})

    def load(self, path: Path) -> LoadedDocument:
        """读取TXT文件并转换为统一文档对象。"""

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
                "The document is not valid UTF-8 text.",
            ) from exc

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text.strip():
            raise EmptyDocumentError(validated_path)

        source_path = validated_path.as_posix()

        return LoadedDocument(
            document_id=validated_path.as_uri(),
            content_hash=sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                source_path=source_path,
                file_name=validated_path.name,
                document_type=DocumentType.TEXT,
                file_size=stat.st_size,
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=UTC,
                ),
                title=validated_path.stem,
                extra={
                    "encoding": "utf-8",
                },
            ),
            parts=[
                DocumentPart(
                    part_index=0,
                    text=text,
                )
            ],
        )
