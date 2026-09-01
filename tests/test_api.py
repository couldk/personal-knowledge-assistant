from fastapi.testclient import TestClient

from personal_knowledge_assistant.agent import (
    AgentOutcome,
    KnowledgeAgentRequest,
    KnowledgeAgentResult,
)
from personal_knowledge_assistant.answering import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.api import (
    create_api_app,
)
from personal_knowledge_assistant.application import (
    DocumentImportOutcome,
)
from personal_knowledge_assistant.domain import (
    ImportStatus,
)
from personal_knowledge_assistant.indexing import (
    IndexingStatus,
)


class FakeAgentService:
    """API 测试使用的 Agent Service。"""

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self._error = error
        self.requests: list[KnowledgeAgentRequest] = []
        self.thread_ids: list[str] = []
        self.histories: dict[
            str,
            list[str],
        ] = {}

    async def run(
        self,
        request: KnowledgeAgentRequest,
        *,
        thread_id: str,
    ) -> KnowledgeAgentResult:
        if self._error is not None:
            raise self._error

        self.requests.append(request)
        self.thread_ids.append(thread_id)

        self.histories.setdefault(
            thread_id,
            [],
        ).append(request.question)

        return KnowledgeAgentResult(
            outcome=AgentOutcome.REFUSED,
            answer=EvidenceAnswer(
                answer=("当前知识库中没有足够证据回答该问题。"),
                citations=[],
                confidence=0.0,
                refused=True,
            ),
            retrieval_attempts=1,
            final_retrieval_query=(request.question),
            evidence=[],
            failure_reason=("Retrieved evidence was insufficient."),
            error_type=None,
        )

    async def get_question_history(
        self,
        *,
        thread_id: str,
    ) -> list[str]:
        if self._error is not None:
            raise self._error

        return list(
            self.histories.get(
                thread_id,
                [],
            )
        )


class FakeDocumentImportService:
    """API测试使用的文档导入服务。"""

    max_upload_bytes = 1024

    def __init__(
        self,
        *,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.uploads: list[tuple[str, bytes]] = []

    async def import_upload(
        self,
        *,
        file_name: str,
        content: bytes,
    ) -> DocumentImportOutcome:
        if self.error is not None:
            raise self.error

        self.uploads.append(
            (
                file_name,
                content,
            )
        )

        return DocumentImportOutcome(
            import_status=ImportStatus.CREATED,
            indexing_status=(IndexingStatus.INDEXED),
            document_id=("file:///uploads/knowledge.txt"),
            content_hash="a" * 64,
            file_name=file_name,
            chunk_count=2,
            deactivated_chunk_count=0,
        )


def test_health_endpoints_return_ok() -> None:
    service = FakeAgentService()
    app = create_api_app(
        agent_service=service,
    )

    with TestClient(app) as client:
        live_response = client.get(
            "/health/live",
        )
        ready_response = client.get(
            "/health/ready",
        )

    assert live_response.status_code == 200
    assert live_response.json() == {
        "status": "ok",
        "service": ("personal-knowledge-assistant"),
    }

    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ok",
        "service": ("personal-knowledge-assistant"),
    }


def test_agent_query_forwards_valid_request() -> None:
    service = FakeAgentService()
    app = create_api_app(
        agent_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/query",
            json={
                "thread_id": "session-1",
                "question": "什么是RAG？",
                "top_k": 3,
                "filters": {
                    "file_name": "rag.md",
                },
                "max_retrieval_attempts": 2,
                "min_relevance_score": 0.35,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["outcome"] == "refused"
    assert payload["answer"]["refused"] is True
    assert payload["retrieval_attempts"] == 1
    assert payload["final_retrieval_query"] == ("什么是RAG？")

    assert len(service.requests) == 1
    assert service.requests[0].question == ("什么是RAG？")
    assert service.requests[0].top_k == 3
    assert service.thread_ids == [
        "session-1",
    ]


def test_agent_query_rejects_invalid_input() -> None:
    service = FakeAgentService()
    app = create_api_app(
        agent_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/query",
            json={
                "thread_id": " ",
                "question": "",
                "top_k": 0,
            },
        )

    assert response.status_code == 422
    assert service.requests == []


def test_agent_history_returns_saved_questions() -> None:
    service = FakeAgentService()
    service.histories["session-1"] = [
        "什么是RAG？",
        "RAG如何减少幻觉？",
    ]

    app = create_api_app(
        agent_service=service,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/agent/threads/session-1/history")

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": "session-1",
        "questions": [
            "什么是RAG？",
            "RAG如何减少幻觉？",
        ],
    }


def test_agent_query_returns_safe_service_error() -> None:
    service = FakeAgentService(
        error=RuntimeError(
            "secret provider error",
        ),
    )

    app = create_api_app(
        agent_service=service,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/query",
            json={
                "thread_id": "session-1",
                "question": "什么是RAG？",
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": ("Knowledge agent is temporarily unavailable."),
    }

    assert "secret provider error" not in (response.text)


def test_document_import_uploads_file() -> None:
    agent_service = FakeAgentService()
    import_service = FakeDocumentImportService()

    app = create_api_app(
        agent_service=agent_service,
        document_import_service=(import_service),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents/import",
            files={
                "file": (
                    "knowledge.txt",
                    b"Vector databases support search.",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "document_id": ("file:///uploads/knowledge.txt"),
        "file_name": "knowledge.txt",
        "content_hash": "a" * 64,
        "import_status": "created",
        "indexing_status": "indexed",
        "chunk_count": 2,
        "deactivated_chunk_count": 0,
    }

    assert import_service.uploads == [
        (
            "knowledge.txt",
            b"Vector databases support search.",
        )
    ]


def test_document_import_rejects_large_file() -> None:
    agent_service = FakeAgentService()
    import_service = FakeDocumentImportService()

    app = create_api_app(
        agent_service=agent_service,
        document_import_service=(import_service),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents/import",
            files={
                "file": (
                    "large.txt",
                    b"x" * 1025,
                    "text/plain",
                )
            },
        )

    assert response.status_code == 413
