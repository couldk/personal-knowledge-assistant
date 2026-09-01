from typing import Protocol, cast

from fastapi import (
    HTTPException,
    Request,
    status,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentRequest,
    KnowledgeAgentResult,
)
from personal_knowledge_assistant.application import (
    DocumentImportOutcome,
)


class AgentServiceProtocol(Protocol):
    """API层依赖的最小Agent Service接口。"""

    async def run(
        self,
        request: KnowledgeAgentRequest,
        *,
        thread_id: str,
    ) -> KnowledgeAgentResult:
        """执行一次Agent查询。"""
        ...

    async def get_question_history(
        self,
        *,
        thread_id: str,
    ) -> list[str]:
        """读取指定会话的问题历史。"""
        ...


class DocumentImportServiceProtocol(Protocol):
    """API层依赖的文档导入服务接口。"""

    @property
    def max_upload_bytes(self) -> int:
        """返回允许上传的最大字节数。"""
        ...

    async def import_upload(
        self,
        *,
        file_name: str,
        content: bytes,
    ) -> DocumentImportOutcome:
        """上传、解析并索引文档。"""
        ...


def get_agent_service(
    request: Request,
) -> AgentServiceProtocol:
    """从FastAPI应用状态读取Agent Service。"""

    service = getattr(
        request.app.state,
        "agent_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Knowledge agent is not ready."),
        )

    return cast(
        AgentServiceProtocol,
        service,
    )


def get_document_import_service(
    request: Request,
) -> DocumentImportServiceProtocol:
    """从FastAPI应用状态读取文档导入服务。"""

    service = getattr(
        request.app.state,
        "document_import_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Document import service is not ready."),
        )

    return cast(
        DocumentImportServiceProtocol,
        service,
    )
