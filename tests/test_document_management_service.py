from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from personal_knowledge_assistant.application import (
    DocumentImportOutcome,
)
from personal_knowledge_assistant.application.document_management import (
    DocumentManagementService,
    DocumentNotFoundError,
)
from personal_knowledge_assistant.domain import (
    DocumentListResult,
    DocumentStatus,
    DocumentType,
    ImportStatus,
    StoredDocument,
)
from personal_knowledge_assistant.indexing import IndexingStatus


def make_document(
    source_path: Path,
    *,
    status: DocumentStatus = DocumentStatus.READY,
) -> StoredDocument:
    now = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)

    return StoredDocument(
        document_key=uuid4(),
        tenant_id="local",
        document_id=source_path.resolve().as_uri(),
        source_path=source_path.resolve().as_posix(),
        file_name=source_path.name,
        document_type=DocumentType.TEXT,
        title=source_path.stem,
        status=status,
        active_content_hash="a" * 64 if status is DocumentStatus.READY else None,
        active_chunk_count=2 if status is DocumentStatus.READY else 0,
        created_at=now,
        updated_at=now,
    )


class FakeDocumentStore:
    def __init__(self, documents: list[StoredDocument]) -> None:
        self.documents = {document.document_key: document for document in documents}

    async def get_document_by_id(
        self,
        document_id: str,
    ) -> StoredDocument | None:
        return next(
            (
                document
                for document in self.documents.values()
                if document.document_id == document_id
            ),
            None,
        )

    async def get_document(
        self,
        document_key: UUID,
    ) -> StoredDocument | None:
        document = self.documents.get(document_key)
        return document.model_copy(deep=True) if document is not None else None

    async def list_documents(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool = False,
    ) -> DocumentListResult:
        documents = [
            document.model_copy(deep=True)
            for document in self.documents.values()
            if include_deleted or document.status is not DocumentStatus.DELETED
        ]

        return DocumentListResult(
            items=documents[offset : offset + limit],
            total=len(documents),
            limit=limit,
            offset=offset,
        )

    async def delete_document(
        self,
        document_key: UUID,
    ) -> bool:
        document = self.documents.get(document_key)

        if document is None or document.status is DocumentStatus.DELETED:
            return False

        self.documents[document_key] = document.model_copy(
            update={
                "status": DocumentStatus.DELETED,
                "active_content_hash": None,
                "active_chunk_count": 0,
            }
        )
        return True


class FakeDocumentReindexer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, bool]] = []

    async def import_upload(
        self,
        *,
        file_name: str,
        content: bytes,
        force_reindex: bool = False,
    ) -> DocumentImportOutcome:
        self.calls.append((file_name, content, force_reindex))

        return DocumentImportOutcome(
            import_status=ImportStatus.UNCHANGED,
            indexing_status=IndexingStatus.INDEXED,
            document_id="file:///uploads/notes.txt",
            content_hash="a" * 64,
            file_name=file_name,
            chunk_count=2,
            deactivated_chunk_count=0,
        )


@pytest.mark.asyncio
async def test_list_and_get_documents(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Vector search.", encoding="utf-8")
    document = make_document(source)
    service = DocumentManagementService(
        document_store=FakeDocumentStore([document]),
        document_import_service=FakeDocumentReindexer(),
    )

    result = await service.list_documents(limit=20, offset=0)
    fetched = await service.get_document(document.document_key)

    assert result.total == 1
    assert result.items == [document]
    assert fetched == document


@pytest.mark.asyncio
async def test_deleted_or_missing_document_is_not_found(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Vector search.", encoding="utf-8")
    deleted = make_document(source, status=DocumentStatus.DELETED)
    service = DocumentManagementService(
        document_store=FakeDocumentStore([deleted]),
        document_import_service=FakeDocumentReindexer(),
    )

    with pytest.raises(DocumentNotFoundError):
        await service.get_document(deleted.document_key)

    with pytest.raises(DocumentNotFoundError):
        await service.get_document(uuid4())


@pytest.mark.asyncio
async def test_delete_document_is_idempotently_reported(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Vector search.", encoding="utf-8")
    document = make_document(source)
    store = FakeDocumentStore([document])
    service = DocumentManagementService(
        document_store=store,
        document_import_service=FakeDocumentReindexer(),
    )

    await service.delete_document(document.document_key)

    with pytest.raises(DocumentNotFoundError):
        await service.delete_document(document.document_key)


@pytest.mark.asyncio
async def test_reindex_reads_source_and_forces_indexing(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Vector search.", encoding="utf-8")
    document = make_document(source)
    reindexer = FakeDocumentReindexer()
    service = DocumentManagementService(
        document_store=FakeDocumentStore([document]),
        document_import_service=reindexer,
    )

    outcome = await service.reindex_document(document.document_key)

    assert outcome.indexing_status is IndexingStatus.INDEXED
    assert reindexer.calls == [
        (
            "notes.txt",
            b"Vector search.",
            True,
        )
    ]


@pytest.mark.asyncio
async def test_reindex_rejects_missing_source(tmp_path: Path) -> None:
    document = make_document(tmp_path / "missing.txt")
    service = DocumentManagementService(
        document_store=FakeDocumentStore([document]),
        document_import_service=FakeDocumentReindexer(),
    )

    with pytest.raises(DocumentNotFoundError):
        await service.reindex_document(document.document_key)
