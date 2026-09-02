from asyncio import to_thread
from pathlib import Path
from typing import Protocol
from uuid import UUID

from personal_knowledge_assistant.application.document_import import (
    DocumentImportOutcome,
)
from personal_knowledge_assistant.domain import (
    DocumentListResult,
    DocumentStatus,
    StoredDocument,
)
from personal_knowledge_assistant.providers.base import (
    DocumentStoreProvider,
)


class DocumentNotFoundError(Exception):
    """文档不存在或已经删除。"""


class DocumentReindexer(Protocol):
    """文档管理服务重新索引时依赖的最小接口。"""

    async def import_upload(
        self,
        *,
        file_name: str,
        content: bytes,
        force_reindex: bool = False,
    ) -> DocumentImportOutcome:
        """导入或强制重新索引上传文档。"""

        ...


class DocumentManagementService:
    def __init__(
        self,
        *,
        document_store: DocumentStoreProvider,
        document_import_service: DocumentReindexer,
    ) -> None:
        self._document_store = document_store
        self._document_import_service = document_import_service

    async def list_documents(
        self,
        *,
        limit: int,
        offset: int,
    ) -> DocumentListResult:
        return await self._document_store.list_documents(
            limit=limit,
            offset=offset,
            include_deleted=False,
        )

    async def get_document(
        self,
        document_key: UUID,
    ) -> StoredDocument:
        document = await self._document_store.get_document(document_key)

        if document is None or document.status is DocumentStatus.DELETED:
            raise DocumentNotFoundError("Document does not exist.")

        return document

    async def delete_document(
        self,
        document_key: UUID,
    ) -> None:
        deleted = await self._document_store.delete_document(document_key)

        if not deleted:
            raise DocumentNotFoundError("Document does not exist.")

    async def reindex_document(
        self,
        document_key: UUID,
    ) -> DocumentImportOutcome:
        document = await self.get_document(document_key)
        source_path = Path(document.source_path)

        if not await to_thread(source_path.is_file):
            raise DocumentNotFoundError("Document source file no longer exists.")

        content = await to_thread(source_path.read_bytes)

        return await self._document_import_service.import_upload(
            file_name=document.file_name,
            content=content,
            force_reindex=True,
        )
