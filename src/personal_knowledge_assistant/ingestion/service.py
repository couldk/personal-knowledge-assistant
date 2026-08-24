from pathlib import Path

from personal_knowledge_assistant.domain import (
    ImportResult,
    ImportStatus,
)
from personal_knowledge_assistant.ingestion.catalog import (
    DocumentCatalog,
)
from personal_knowledge_assistant.ingestion.registry import LoaderRegistry


class DocumentIngestionService:
    """加载文档并判断创建、重复或更新状态。"""

    def __init__(
        self,
        registry: LoaderRegistry,
        catalog: DocumentCatalog,
    ) -> None:
        self._registry = registry
        self._catalog = catalog

    def import_document(
        self,
        path: Path,
    ) -> ImportResult:
        """导入文档，并与已经保存的版本进行比较。"""

        document = self._registry.load(path)
        existing_document = self._catalog.get(
            document.document_id,
        )

        if existing_document is None:
            self._catalog.save(document)

            return ImportResult(
                status=ImportStatus.CREATED,
                document=document,
            )

        if existing_document.content_hash == document.content_hash:
            return ImportResult(
                status=ImportStatus.UNCHANGED,
                document=existing_document,
                previous_content_hash=existing_document.content_hash,
            )

        previous_content_hash = existing_document.content_hash
        self._catalog.save(document)

        return ImportResult(
            status=ImportStatus.UPDATED,
            document=document,
            previous_content_hash=previous_content_hash,
        )
