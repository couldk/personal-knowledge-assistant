from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from personal_knowledge_assistant.domain import LoadedDocument
from personal_knowledge_assistant.ingestion.exceptions import (
    DocumentNotFoundError,
    InvalidDocumentPathError,
)


def normalize_extension(extension: str) -> str:
    """将扩展名规范为小写且以点开头。"""

    normalized = extension.strip().casefold()

    if not normalized:
        return ""

    if not normalized.startswith("."):
        return f".{normalized}"

    return normalized


def validate_source_path(path: Path) -> Path:
    """验证文档路径，并返回规范化的绝对路径。"""

    try:
        resolved_path = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise DocumentNotFoundError(path) from exc

    if not resolved_path.is_file():
        raise InvalidDocumentPathError(resolved_path)

    return resolved_path


class DocumentLoader(ABC):
    """所有文档Loader必须实现的统一接口。"""

    supported_extensions: ClassVar[frozenset[str]] = frozenset()

    def supports(self, path: Path) -> bool:
        """判断当前Loader是否支持指定文件。"""

        extension = normalize_extension(path.suffix)

        return extension in {normalize_extension(item) for item in self.supported_extensions}

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument:
        """读取并解析文件，返回统一文档对象。"""
