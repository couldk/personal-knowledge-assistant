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


class AgentServiceProtocol(Protocol):
    """API 层依赖的最小 Agent Service 接口。"""

    async def run(
        self,
        request: KnowledgeAgentRequest,
        *,
        thread_id: str,
    ) -> KnowledgeAgentResult:
        """执行一次 Agent 查询。"""
        ...

    async def get_question_history(
        self,
        *,
        thread_id: str,
    ) -> list[str]:
        """读取指定会话的问题历史。"""
        ...


def get_agent_service(
    request: Request,
) -> AgentServiceProtocol:
    """从 FastAPI 应用状态读取 Agent Service。"""

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
