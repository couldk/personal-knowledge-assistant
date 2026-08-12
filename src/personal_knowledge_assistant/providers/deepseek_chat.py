from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from openai.types.chat import ChatCompletionMessageParam

from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from personal_knowledge_assistant.providers.exceptions import (
    ChatProviderError,
    EmptyChatMessagesError,
    EmptyChatResponseError,
)


class DeepSeekChatProvider:
    """Chat provider backed by DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        if not messages or any(not message.content.strip() for message in messages):
            raise EmptyChatMessagesError("At least one non-empty chat message is required.")

        request_messages = self._convert_messages(messages)

        try:
            if json_mode:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )
            else:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
        except (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
        ) as exc:
            raise ChatProviderError(f"DeepSeek chat request failed: {exc}") from exc

        if not response.choices:
            raise EmptyChatResponseError("DeepSeek returned no response choices.")

        content = response.choices[0].message.content

        if content is None or not content.strip():
            raise EmptyChatResponseError("DeepSeek returned empty response content.")

        usage: dict[str, int] = {}

        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=content.strip(),
            model=response.model or self._model,
            usage=usage,
        )

    @staticmethod
    def _convert_messages(
        messages: Sequence[ChatMessage],
    ) -> list[ChatCompletionMessageParam]:
        request_messages: list[ChatCompletionMessageParam] = []

        for message in messages:
            if message.role is MessageRole.SYSTEM:
                request_messages.append(
                    {
                        "role": "system",
                        "content": message.content,
                    }
                )
            elif message.role is MessageRole.USER:
                request_messages.append(
                    {
                        "role": "user",
                        "content": message.content,
                    }
                )
            else:
                request_messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )

        return request_messages
