from personal_knowledge_assistant.answering.prompt_builder import (
    EvidencePromptBuilder,
)
from personal_knowledge_assistant.answering.service import (
    AnsweringService,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
)
from personal_knowledge_assistant.providers.factory import (
    create_chat_provider,
)


def create_answering_service(
    *,
    settings: Settings,
    chat_provider: ChatProvider | None = None,
) -> AnsweringService:
    """根据应用配置创建回答服务。"""

    selected_chat_provider = (
        chat_provider if chat_provider is not None else create_chat_provider(settings)
    )

    return AnsweringService(
        chat_provider=selected_chat_provider,
        prompt_builder=EvidencePromptBuilder(),
    )
