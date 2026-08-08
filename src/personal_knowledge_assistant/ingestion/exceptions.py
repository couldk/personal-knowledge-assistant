from pathlib import Path


class IngestionError(Exception):
    """所有文档导入异常的基类。"""


class DocumentNotFoundError(IngestionError):
    """文档路径不存在。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Document does not exist: {path}")


class InvalidDocumentPathError(IngestionError):
    """路径存在，但不是普通文件。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Document path is not a file: {path}")


class UnsupportedDocumentTypeError(IngestionError):
    """没有找到支持该文件类型的Loader。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        extension = path.suffix or "<no extension>"
        super().__init__(f"Unsupported document type: {extension}")


class DuplicateLoaderRegistrationError(IngestionError):
    """同一扩展名被多个Loader重复注册。"""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(f"A loader is already registered for: {extension}")


class DocumentReadError(IngestionError):
    """文件存在，但读取失败。"""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to read document {path}: {reason}")


class EmptyDocumentError(IngestionError):
    """文件存在，但没有可导入的文本内容。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Document contains no readable text: {path}")


class DocumentParseError(IngestionError):
    """文件能够读取，但内容无法解析。"""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse document {path}: {reason}")
