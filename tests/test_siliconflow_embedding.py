from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import Request
from openai import APITimeoutError, AsyncOpenAI

from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.providers.base import EmbeddingProvider
from personal_knowledge_assistant.providers.exceptions import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    EmptyEmbeddingInputError,
)
from personal_knowledge_assistant.providers.factory import (
    create_embedding_provider,
)
from personal_knowledge_assistant.providers.siliconflow_embedding import (
    SiliconFlowEmbeddingProvider,
)


def _response(
    *indexed_vectors: tuple[int, list[float]],
) -> SimpleNamespace:
    """创建模拟的 Embedding API 响应。"""

    return SimpleNamespace(
        data=[
            SimpleNamespace(
                index=index,
                embedding=embedding,
            )
            for index, embedding in indexed_vectors
        ]
    )


def _make_client(
    *responses: SimpleNamespace,
) -> tuple[AsyncOpenAI, AsyncMock]:
    """创建不会访问网络的模拟 AsyncOpenAI 客户端。"""

    create_mock = AsyncMock(
        side_effect=list(responses),
    )
    raw_client = MagicMock()
    raw_client.embeddings.create = create_mock

    return cast(AsyncOpenAI, raw_client), create_mock


def _make_provider(
    client: AsyncOpenAI,
    *,
    dimension: int = 3,
    batch_size: int = 32,
) -> SiliconFlowEmbeddingProvider:
    """使用测试配置创建硅基流动 Provider。"""

    return SiliconFlowEmbeddingProvider(
        api_key="test-api-key",
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://api.siliconflow.cn/v1",
        dimension=dimension,
        batch_size=batch_size,
        timeout_seconds=30.0,
        max_retries=2,
        client=client,
    )


@pytest.mark.asyncio
async def test_embed_texts_returns_vectors_in_input_order() -> None:
    client, create_mock = _make_client(
        _response(
            (1, [0.0, 1.0, 0.0]),
            (0, [1.0, 0.0, 0.0]),
        )
    )
    provider = _make_provider(client)

    result = await provider.embed_texts(
        [" first ", "second"],
    )

    assert result == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    create_mock.assert_awaited_once_with(
        model="Qwen/Qwen3-Embedding-0.6B",
        input=["first", "second"],
        dimensions=3,
        encoding_format="float",
    )


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector() -> None:
    client, _ = _make_client(
        _response(
            (0, [0.1, 0.2, 0.3]),
        )
    )
    provider = _make_provider(client)

    result = await provider.embed_query(
        "个人知识助手",
    )

    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["   "],
        ["正常文本", ""],
    ],
)
async def test_embed_texts_rejects_empty_input(
    texts: list[str],
) -> None:
    client, create_mock = _make_client()
    provider = _make_provider(client)

    with pytest.raises(EmptyEmbeddingInputError):
        await provider.embed_texts(texts)

    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_texts_rejects_wrong_dimension() -> None:
    client, _ = _make_client(
        _response(
            (0, [0.1, 0.2]),
        )
    )
    provider = _make_provider(
        client,
        dimension=3,
    )

    with pytest.raises(
        EmbeddingDimensionError,
        match="Expected 3 dimensions",
    ):
        await provider.embed_texts(["test"])


@pytest.mark.asyncio
async def test_embed_texts_rejects_missing_response_item() -> None:
    client, _ = _make_client(
        _response(
            (0, [1.0, 0.0, 0.0]),
        )
    )
    provider = _make_provider(client)

    with pytest.raises(
        EmbeddingProviderError,
        match="Expected 2 embeddings",
    ):
        await provider.embed_texts(
            ["first", "second"],
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_invalid_indexes() -> None:
    client, _ = _make_client(
        _response(
            (0, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
        )
    )
    provider = _make_provider(client)

    with pytest.raises(
        EmbeddingProviderError,
        match="invalid indexes",
    ):
        await provider.embed_texts(
            ["first", "second"],
        )


@pytest.mark.asyncio
async def test_embed_texts_splits_large_input_into_batches() -> None:
    client, create_mock = _make_client(
        _response(
            (0, [1.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
        ),
        _response(
            (0, [0.0, 0.0, 1.0]),
            (1, [1.0, 1.0, 0.0]),
        ),
        _response(
            (0, [1.0, 0.0, 1.0]),
        ),
    )
    provider = _make_provider(
        client,
        batch_size=2,
    )

    result = await provider.embed_texts(
        [
            "text 1",
            "text 2",
            "text 3",
            "text 4",
            "text 5",
        ]
    )

    assert len(result) == 5
    assert create_mock.await_count == 3

    submitted_batches = [call.kwargs["input"] for call in create_mock.await_args_list]

    assert submitted_batches == [
        ["text 1", "text 2"],
        ["text 3", "text 4"],
        ["text 5"],
    ]


@pytest.mark.asyncio
async def test_embed_texts_converts_timeout_error() -> None:
    client, create_mock = _make_client()
    create_mock.side_effect = APITimeoutError(
        request=Request(
            "POST",
            "https://api.siliconflow.cn/v1/embeddings",
        )
    )
    provider = _make_provider(client)

    with pytest.raises(
        EmbeddingProviderError,
        match="SiliconFlow embedding request failed",
    ):
        await provider.embed_texts(["test"])


def test_provider_implements_embedding_protocol() -> None:
    client, _ = _make_client()
    provider = _make_provider(client)

    assert isinstance(provider, EmbeddingProvider)


def test_factory_creates_siliconflow_provider() -> None:
    settings = Settings.model_validate(
        {
            "embedding_provider": "siliconflow",
            "embedding_model": ("Qwen/Qwen3-Embedding-0.6B"),
            "embedding_api_key": "test-api-key",
            "embedding_base_url": ("https://api.siliconflow.cn/v1"),
            "embedding_dimension": 1024,
        }
    )

    provider = create_embedding_provider(settings)

    assert isinstance(
        provider,
        SiliconFlowEmbeddingProvider,
    )


def test_factory_rejects_missing_api_key() -> None:
    settings = Settings.model_validate(
        {
            "embedding_provider": "siliconflow",
            "embedding_model": ("Qwen/Qwen3-Embedding-0.6B"),
            "embedding_api_key": None,
            "embedding_dimension": 1024,
        }
    )

    with pytest.raises(
        ValueError,
        match="Embedding API key is required",
    ):
        create_embedding_provider(settings)


def test_factory_rejects_missing_dimension() -> None:
    settings = Settings.model_validate(
        {
            "embedding_provider": "siliconflow",
            "embedding_model": ("Qwen/Qwen3-Embedding-0.6B"),
            "embedding_api_key": "test-api-key",
            "embedding_dimension": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimension is required",
    ):
        create_embedding_provider(settings)


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings.model_validate(
        {
            "embedding_provider": "unknown",
            "embedding_model": "unknown",
            "embedding_api_key": "test-api-key",
            "embedding_dimension": 1024,
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider",
    ):
        create_embedding_provider(settings)
