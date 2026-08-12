class ProviderError(Exception):
    """提供商调用失败。"""


class EmbeddingProviderError(ProviderError):
    """Embedding 提供商调用失败。"""


class EmptyEmbeddingInputError(EmbeddingProviderError):
    """Embedding 输入为空。"""


class EmbeddingDimensionError(EmbeddingProviderError):
    """Embedding 返回的向量维度不正确。"""


class ChatProviderError(ProviderError):
    """Base exception raised by chat providers."""


class EmptyChatMessagesError(ChatProviderError):
    """Raised when a chat request contains no usable messages."""


class EmptyChatResponseError(ChatProviderError):
    """Raised when the chat provider returns no usable content."""
