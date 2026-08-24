from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Request
from openai import APITimeoutError, AsyncOpenAI

from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    MessageRole,
)
from personal_knowledge_assistant.providers.base import ChatProvider
from personal_knowledge_assistant.providers.deepseek_chat import (
    DeepSeekChatProvider,
)
from personal_knowledge_assistant.providers.exceptions import (
    ChatProviderError,
    EmptyChatMessagesError,
    EmptyChatResponseError,
)
from personal_knowledge_assistant.providers.factory import (
    create_chat_provider,
)


def _make_client(
    *responses: object,
) -> tuple[AsyncOpenAI, AsyncMock]:
    """创建不会访问网络的模拟 AsyncOpenAI 客户端。"""

    create_mock = AsyncMock(side_effect=list(responses))
    raw_client = MagicMock()
    raw_client.chat.completions.create = create_mock
    return cast(AsyncOpenAI, raw_client), create_mock


def _make_response(
    content: str | None = "Hello",
    *,
    choices: bool = True,
    usage: bool = True,
) -> SimpleNamespace:
    """创建模拟的 Chat Completion API 响应。"""

    return SimpleNamespace(
        choices=([SimpleNamespace(message=SimpleNamespace(content=content))] if choices else []),
        model="deepseek-v4-flash",
        usage=(
            SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
            if usage
            else None
        ),
    )


def _make_provider(client: AsyncOpenAI) -> DeepSeekChatProvider:
    """使用测试配置创建 DeepSeek Provider。"""

    return DeepSeekChatProvider(
        api_key="test-api-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout_seconds=60.0,
        max_retries=2,
        max_tokens=2048,
        temperature=0.0,
        client=client,
    )


@pytest.mark.asyncio
async def test_complete_returns_chat_response_and_usage() -> None:
    client, create_mock = _make_client(_make_response(" Test answer "))
    provider = _make_provider(client)

    response = await provider.complete([ChatMessage(role=MessageRole.USER, content="Hello")])

    assert response.content == "Test answer"
    assert response.model == "deepseek-v4-flash"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    create_mock.assert_awaited_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=2048,
        temperature=0.0,
    )


@pytest.mark.asyncio
async def test_complete_maps_all_message_roles() -> None:
    client, create_mock = _make_client(_make_response())
    provider = _make_provider(client)

    await provider.complete(
        [
            ChatMessage(role=MessageRole.SYSTEM, content="System instruction"),
            ChatMessage(role=MessageRole.USER, content="Question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Earlier answer"),
        ]
    )

    assert create_mock.await_args_list[-1].kwargs["messages"] == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]


@pytest.mark.asyncio
async def test_complete_enables_json_mode() -> None:
    client, create_mock = _make_client(_make_response('{"answer":"ok"}'))
    provider = _make_provider(client)

    response = await provider.complete(
        [
            ChatMessage(
                role=MessageRole.USER,
                content="Return a JSON object.",
            )
        ],
        json_mode=True,
    )

    assert response.content == '{"answer":"ok"}'
    assert create_mock.await_args_list[-1].kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_complete_allows_missing_usage() -> None:
    client, _ = _make_client(_make_response(usage=False))
    provider = _make_provider(client)

    response = await provider.complete([ChatMessage(role=MessageRole.USER, content="Hello")])

    assert response.usage == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [ChatMessage(role=MessageRole.USER, content="   ")],
    ],
)
async def test_complete_rejects_empty_messages(
    messages: list[ChatMessage],
) -> None:
    client, create_mock = _make_client()
    provider = _make_provider(client)

    with pytest.raises(EmptyChatMessagesError):
        await provider.complete(messages)

    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_rejects_missing_choices() -> None:
    client, _ = _make_client(_make_response(choices=False))
    provider = _make_provider(client)

    with pytest.raises(
        EmptyChatResponseError,
        match="no response choices",
    ):
        await provider.complete([ChatMessage(role=MessageRole.USER, content="Hello")])


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, "", "   "])
async def test_complete_rejects_empty_response_content(
    content: str | None,
) -> None:
    client, _ = _make_client(_make_response(content))
    provider = _make_provider(client)

    with pytest.raises(
        EmptyChatResponseError,
        match="empty response content",
    ):
        await provider.complete([ChatMessage(role=MessageRole.USER, content="Hello")])


@pytest.mark.asyncio
async def test_complete_converts_timeout_error() -> None:
    client, create_mock = _make_client()
    create_mock.side_effect = APITimeoutError(
        request=Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )
    )
    provider = _make_provider(client)

    with pytest.raises(
        ChatProviderError,
        match="DeepSeek chat request failed",
    ):
        await provider.complete([ChatMessage(role=MessageRole.USER, content="Hello")])


def test_provider_implements_chat_protocol() -> None:
    client, _ = _make_client()
    provider = _make_provider(client)

    assert isinstance(provider, ChatProvider)


def _chat_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "chat_provider": "deepseek",
        "chat_model": "deepseek-v4-flash",
        "chat_api_key": "test-api-key",
        "chat_base_url": "https://api.deepseek.com",
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_factory_creates_deepseek_provider() -> None:
    provider = create_chat_provider(_chat_settings())

    assert isinstance(provider, DeepSeekChatProvider)


def test_factory_rejects_missing_api_key() -> None:
    with pytest.raises(ValueError, match="Chat API key is required"):
        create_chat_provider(_chat_settings(chat_api_key=None))


@pytest.mark.parametrize("chat_model", ["", "   ", "replace-me"])
def test_factory_rejects_missing_model(chat_model: str) -> None:
    with pytest.raises(ValueError, match="Chat model is required"):
        create_chat_provider(_chat_settings(chat_model=chat_model))


def test_factory_rejects_missing_base_url() -> None:
    with pytest.raises(ValueError, match="Chat API base URL is required"):
        create_chat_provider(_chat_settings(chat_base_url="   "))


def test_factory_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported chat provider"):
        create_chat_provider(_chat_settings(chat_provider="unknown"))
