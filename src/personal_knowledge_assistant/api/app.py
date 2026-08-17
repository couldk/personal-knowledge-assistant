from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from personal_knowledge_assistant.api.dependencies import (
    AgentServiceProtocol,
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


def create_api_app(
    settings: Settings | None = None,
    *,
    agent_service: (AgentServiceProtocol | None) = None,
) -> FastAPI:
    """创建个人知识助手 FastAPI 应用。

    测试可以注入 Fake Agent Service，
    生产运行则由应用工厂创建真实服务。
    """

    selected_settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(
        api: FastAPI,
    ) -> AsyncIterator[None]:
        """在应用启动时只创建一次共享服务。"""

        selected_agent_service = agent_service

        if selected_agent_service is None:
            services = create_application_services(
                selected_settings,
            )
            selected_agent_service = services.agent_service

        api.state.agent_service = selected_agent_service

        yield

    api = FastAPI(
        title="Personal Knowledge Assistant API",
        description=("Evidence-grounded LangGraph personal knowledge assistant."),
        version="0.1.0",
        lifespan=lifespan,
    )

    api.include_router(
        router,
    )

    return api


app = create_api_app()
