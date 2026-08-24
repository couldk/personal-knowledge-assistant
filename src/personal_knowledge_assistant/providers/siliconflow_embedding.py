from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from personal_knowledge_assistant.providers.exceptions import (
    EmbeddingDimensionError,
    EmbeddingProviderError,
    EmptyEmbeddingInputError,
)


class SiliconFlowEmbeddingProvider:
    """通过 OpenAI 兼容接口调用硅基流动 Embedding。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        dimension: int,
        batch_size: int = 32,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        normalized = [text.strip() for text in texts]

        if not normalized or any(not text for text in normalized):
            raise EmptyEmbeddingInputError("Embedding input must contain non-empty text.")

        result: list[list[float]] = []

        try:
            for start in range(0, len(normalized), self._batch_size):
                batch = normalized[start : start + self._batch_size]

                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimension,
                    encoding_format="float",
                )

                data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                if len(data) != len(batch):
                    raise EmbeddingProviderError(
                        f"Expected {len(batch)} embeddings, received {len(data)}."
                    )

                expected_indexes = list(range(len(batch)))
                actual_indexes = [item.index for item in data]

                if actual_indexes != expected_indexes:
                    raise EmbeddingProviderError("Embedding response contains invalid indexes.")

                for item in data:
                    embedding = list(item.embedding)

                    if len(embedding) != self._dimension:
                        raise EmbeddingDimensionError(
                            f"Expected {self._dimension} dimensions, received {len(embedding)}."
                        )

                    result.append(embedding)

        except EmbeddingDimensionError:
            raise
        except (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        ) as exc:
            raise EmbeddingProviderError("SiliconFlow embedding request failed.") from exc

        return result

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        embeddings = await self.embed_texts([query])
        return embeddings[0]
