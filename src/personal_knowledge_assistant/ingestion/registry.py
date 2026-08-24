from collections.abc import Iterable
from pathlib import Path

from personal_knowledge_assistant.domain import LoadedDocument
from personal_knowledge_assistant.ingestion.base import (
    DocumentLoader,
    normalize_extension,
    validate_source_path,
)
from personal_knowledge_assistant.ingestion.exceptions import (
    DuplicateLoaderRegistrationError,
    UnsupportedDocumentTypeError,
)


class LoaderRegistry:
    """管理文件扩展名与Loader之间的映射。"""

    def __init__(
        self,
        loaders: Iterable[DocumentLoader] = (),
    ) -> None:
        self._loaders: dict[str, DocumentLoader] = {}

        for loader in loaders:
            self.register(loader)

    def register(self, loader: DocumentLoader) -> None:
        """注册一个文档Loader。"""

        if not loader.supported_extensions:
            raise ValueError("A document loader must declare supported extensions.")

        normalized_extensions = {
            normalize_extension(extension) for extension in loader.supported_extensions
        }

        if "" in normalized_extensions:
            raise ValueError("A document extension cannot be empty.")

        for extension in normalized_extensions:
            if extension in self._loaders:
                raise DuplicateLoaderRegistrationError(extension)

        for extension in normalized_extensions:
            self._loaders[extension] = loader

    def get_loader(self, path: Path) -> DocumentLoader:
        """根据文件扩展名选择Loader。"""

        extension = normalize_extension(path.suffix)
        loader = self._loaders.get(extension)

        if loader is None:
            raise UnsupportedDocumentTypeError(path)

        return loader

    def load(self, path: Path) -> LoadedDocument:
        """验证路径、选择Loader并读取文档。"""

        validated_path = validate_source_path(path)
        loader = self.get_loader(validated_path)

        return loader.load(validated_path)
