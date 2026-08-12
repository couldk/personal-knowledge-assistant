import asyncio
import json

from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    MessageRole,
)
from personal_knowledge_assistant.providers import create_chat_provider


async def main() -> None:
    """使用最小 JSON 请求验证真实 DeepSeek Chat API。"""

    settings = Settings()
    provider = create_chat_provider(settings)
    response = await provider.complete(
        [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=('Return JSON only using this schema: {"status": "string"}.'),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=("Return a JSON object confirming that the connection works."),
            ),
        ],
        json_mode=True,
    )
    payload = json.loads(response.content)

    print("Real API verification: PASSED")
    print("Model:", response.model)
    print("JSON status:", payload.get("status"))
    print("Usage:", response.usage)


if __name__ == "__main__":
    asyncio.run(main())
