from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from personal_knowledge_assistant.api import create_api_app
from personal_knowledge_assistant.application import DocumentImportOutcome
from personal_knowledge_assistant.application.document_management import (
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


def make_document() -> StoredDocument:
    now = datetime(2026, 9, 2, 10, 0, tzinfo=UTC)

    return StoredDocument(
        document_key=uuid4(),
        tenant_id="local",
        document_id="file:///uploads/notes.txt",
        source_path="D:/uploads/notes.txt",
        file_name="notes.txt",
        document_type=DocumentType.TEXT,
        title="notes",
        status=DocumentStatus.READY,
        active_content_hash="a" * 64,
        active_chunk_count=2,
        created_at=now,
        updated_at=now,
    )


class FakeDocumentManagementService:
    def __init__(
        self,
        document: StoredDocument,
        *,
        error: Exception | None = None,
    ) -> None:
        self.document = document
        self.error = error
        self.deleted_keys: list[UUID] = []
        self.reindexed_keys: list[UUID] = []

    async def list_documents(
        self,
        *,
        limit: int,
        offset: int,
    ) -> DocumentListResult:
        if self.error is not None:
            raise self.error

        return DocumentListResult(
            items=[self.document],
            total=1,
            limit=limit,
            offset=offset,
        )

    async def get_document(
        self,
        document_key: UUID,
    ) -> StoredDocument:
        if self.error is not None:
            raise self.error

        if document_key != self.document.document_key:
            raise DocumentNotFoundError("missing")

        return self.document

    async def delete_document(
        self,
        document_key: UUID,
    ) -> None:
        if self.error is not None:
            raise self.error

        if document_key != self.document.document_key:
            raise DocumentNotFoundError("missing")

        self.deleted_keys.append(document_key)

    async def reindex_document(
        self,
        document_key: UUID,
    ) -> DocumentImportOutcome:
        if self.error is not None:
            raise self.error

        if document_key != self.document.document_key:
            raise DocumentNotFoundError("missing")

        self.reindexed_keys.append(document_key)

        return DocumentImportOutcome(
            import_status=ImportStatus.UNCHANGED,
            indexing_status=IndexingStatus.INDEXED,
            document_id=self.document.document_id,
            content_hash="a" * 64,
            file_name=self.document.file_name,
            chunk_count=2,
            deactivated_chunk_count=0,
        )


def test_document_list_and_detail_endpoints() -> None:
    document = make_document()
    service = FakeDocumentManagementService(document)
    app = create_api_app(document_management_service=service)

    with TestClient(app) as client:
        list_response = client.get("/api/v1/documents?limit=10&offset=0")
        detail_response = client.get(f"/api/v1/documents/{document.document_key}")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert list_response.json()["items"][0]["document_key"] == str(document.document_key)
    assert detail_response.status_code == 200
    assert detail_response.json()["file_name"] == "notes.txt"


def test_document_management_rejects_invalid_pagination() -> None:
    document = make_document()
    app = create_api_app(document_management_service=FakeDocumentManagementService(document))

    with TestClient(app) as client:
        response = client.get("/api/v1/documents?limit=0&offset=-1")

    assert response.status_code == 422


def test_document_detail_returns_404() -> None:
    document = make_document()
    app = create_api_app(document_management_service=FakeDocumentManagementService(document))

    with TestClient(app) as client:
        response = client.get(f"/api/v1/documents/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document was not found."}


def test_document_delete_and_reindex_endpoints() -> None:
    document = make_document()
    service = FakeDocumentManagementService(document)
    app = create_api_app(document_management_service=service)

    with TestClient(app) as client:
        delete_response = client.delete(f"/api/v1/documents/{document.document_key}")
        reindex_response = client.post(f"/api/v1/documents/{document.document_key}/reindex")

    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert service.deleted_keys == [document.document_key]

    assert reindex_response.status_code == 200
    assert reindex_response.json()["indexing_status"] == "indexed"
    assert service.reindexed_keys == [document.document_key]


def test_document_management_returns_safe_service_error() -> None:
    document = make_document()
    service = FakeDocumentManagementService(
        document,
        error=RuntimeError("secret database error"),
    )
    app = create_api_app(document_management_service=service)

    with TestClient(app) as client:
        response = client.get("/api/v1/documents")

    assert response.status_code == 503
    assert response.json() == {"detail": "Document management is temporarily unavailable."}
    assert "secret database error" not in response.text
