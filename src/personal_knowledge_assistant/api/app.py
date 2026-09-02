from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from personal_knowledge_assistant.api.dependencies import (
    AgentServiceProtocol,
    DocumentImportServiceProtocol,
    DocumentManagementServiceProtocol,
)
from personal_knowledge_assistant.api.routes import (
    router,
)
from personal_knowledge_assistant.application import (
    create_application_services,
)
from personal_knowledge_assistant.config import (
    Settings,
)
from personal_knowledge_assistant.vector_store import (
    PgVectorStore,
)


def create_api_app(
    settings: Settings | None = None,
    *,
    agent_service: (AgentServiceProtocol | None) = None,
    document_import_service: (DocumentImportServiceProtocol | None) = None,
    document_management_service: (DocumentManagementServiceProtocol | None) = None,
) -> FastAPI:
    """创建个人知识助手FastAPI应用。

    测试可以注入Fake Service，
    生产运行则由应用工厂创建真实服务。
    """

    selected_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(
        api: FastAPI,
    ) -> AsyncIterator[None]:
        """创建共享服务并管理数据库连接池。"""

        selected_agent_service = agent_service
        selected_document_import_service = document_import_service
        selected_document_management_service = document_management_service

        managed_vector_store: PgVectorStore | None = None

        # 生产环境没有注入Fake Service，
        # 因此创建完整的真实应用服务。
        if (
            selected_agent_service is None
            and selected_document_import_service is None
            and selected_document_management_service is None
        ):
            services = create_application_services(
                selected_settings,
            )

            if isinstance(
                services.vector_store,
                PgVectorStore,
            ):
                await services.vector_store.open()
                managed_vector_store = services.vector_store

            selected_agent_service = services.agent_service
            selected_document_import_service = services.document_import_service
            selected_document_management_service = services.document_management_service

        api.state.agent_service = selected_agent_service
        api.state.document_import_service = selected_document_import_service
        api.state.document_management_service = selected_document_management_service

        try:
            yield
        finally:
            if managed_vector_store is not None:
                await managed_vector_store.close()

    api = FastAPI(
        title=("Personal Knowledge Assistant API"),
        description=("Evidence-grounded LangGraph personal knowledge assistant."),
        version="0.1.0",
        lifespan=lifespan,
    )

    api.include_router(
        router,
    )

    return api


app = create_api_app()
