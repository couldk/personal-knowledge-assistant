import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from personal_knowledge_assistant.agent import (
    KnowledgeAgentResult,
)
from personal_knowledge_assistant.api.dependencies import (
    AgentServiceProtocol,
    DocumentImportServiceProtocol,
    get_agent_service,
    get_document_import_service,
)
from personal_knowledge_assistant.api.models import (
    AgentHistoryResponse,
    AgentQueryRequest,
    ApiErrorResponse,
    DocumentImportResponse,
    HealthResponse,
)
from personal_knowledge_assistant.application import (
    InvalidUploadFileNameError,
    UnsupportedUploadTypeError,
    UploadTooLargeError,
)
from personal_knowledge_assistant.ingestion import (
    IngestionError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

AgentServiceDependency = Annotated[
    AgentServiceProtocol,
    Depends(get_agent_service),
]

DocumentImportServiceDependency = Annotated[
    DocumentImportServiceProtocol,
    Depends(get_document_import_service),
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
    """执行一次知识Agent查询。"""

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


@router.post(
    "/api/v1/documents/import",
    response_model=DocumentImportResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": ApiErrorResponse,
            "description": "Invalid file name.",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "model": ApiErrorResponse,
            "description": ("Uploaded document is too large."),
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "model": ApiErrorResponse,
            "description": ("Unsupported document type."),
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ApiErrorResponse,
            "description": ("Document cannot be parsed."),
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ApiErrorResponse,
            "description": ("Document import is unavailable."),
        },
    },
    tags=["documents"],
)
async def import_document(
    file: Annotated[
        UploadFile,
        File(),
    ],
    document_import_service: (DocumentImportServiceDependency),
) -> DocumentImportResponse:
    """上传、解析并索引一个文档。"""

    file_name = file.filename or ""

    try:
        # 只读取“最大允许大小+1”字节。
        # 如果多出的1字节存在，立即返回413。
        content = await file.read(document_import_service.max_upload_bytes + 1)

        if len(content) > document_import_service.max_upload_bytes:
            raise UploadTooLargeError("Uploaded document exceeds the size limit.")

        outcome = await document_import_service.import_upload(
            file_name=file_name,
            content=content,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_413_CONTENT_TOO_LARGE),
            detail=("Uploaded document is too large."),
        ) from exc
    except UnsupportedUploadTypeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
            detail=("Only TXT, Markdown and PDF are supported."),
        ) from exc
    except InvalidUploadFileNameError as exc:
        raise HTTPException(
            status_code=(status.HTTP_400_BAD_REQUEST),
            detail=("Upload file name is invalid."),
        ) from exc
    except IngestionError as exc:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=("Uploaded document cannot be parsed."),
        ) from exc
    except Exception as exc:
        logger.exception(
            "document_import_failed",
            extra={
                "error_type": type(exc).__name__,
            },
        )

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Document import is temporarily unavailable."),
        ) from exc
    finally:
        await file.close()

    return DocumentImportResponse(
        document_id=outcome.document_id,
        file_name=outcome.file_name,
        content_hash=outcome.content_hash,
        import_status=outcome.import_status,
        indexing_status=outcome.indexing_status,
        chunk_count=outcome.chunk_count,
        deactivated_chunk_count=(outcome.deactivated_chunk_count),
    )
