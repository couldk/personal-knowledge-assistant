import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentResult,
)
from personal_knowledge_assistant.api.dependencies import (
    AgentServiceProtocol,
    get_agent_service,
)
from personal_knowledge_assistant.api.models import (
    AgentHistoryResponse,
    AgentQueryRequest,
    ApiErrorResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

AgentServiceDependency = Annotated[
    AgentServiceProtocol,
    Depends(get_agent_service),
]


@router.get(
    "/health/live",
    response_model=HealthResponse,
    tags=["health"],
)
async def liveness() -> HealthResponse:
    """进程存活检查。"""

    return HealthResponse(
        status="ok",
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=["health"],
)
async def readiness(
    agent_service: AgentServiceDependency,
) -> HealthResponse:
    """应用依赖就绪检查。"""

    del agent_service

    return HealthResponse(
        status="ok",
    )


@router.post(
    "/api/v1/agent/query",
    response_model=KnowledgeAgentResult,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid request.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ApiErrorResponse,
            "description": ("Agent is temporarily unavailable."),
        },
    },
    tags=["agent"],
)
async def query_agent(
    payload: AgentQueryRequest,
    agent_service: AgentServiceDependency,
) -> KnowledgeAgentResult:
    """执行一次知识 Agent 查询。"""

    try:
        return await agent_service.run(
            payload.to_agent_request(),
            thread_id=payload.thread_id,
        )
    except Exception as exc:
        logger.exception(
            "agent_api_query_failed",
            extra={
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Knowledge agent is temporarily unavailable."),
        ) from exc


@router.get(
    "/api/v1/agent/threads/{thread_id}/history",
    response_model=AgentHistoryResponse,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Invalid thread ID.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ApiErrorResponse,
            "description": ("Agent is temporarily unavailable."),
        },
    },
    tags=["agent"],
)
async def get_agent_history(
    thread_id: str,
    agent_service: AgentServiceDependency,
) -> AgentHistoryResponse:
    """读取指定会话的问题历史。"""

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail="thread_id cannot be empty.",
        )

    if len(normalized_thread_id) > 128:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=("thread_id cannot exceed 128 characters."),
        )

    try:
        questions = await agent_service.get_question_history(
            thread_id=normalized_thread_id,
        )
    except Exception as exc:
        logger.exception(
            "agent_api_history_failed",
            extra={
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Knowledge agent is temporarily unavailable."),
        ) from exc

    return AgentHistoryResponse(
        thread_id=normalized_thread_id,
        questions=questions,
    )
