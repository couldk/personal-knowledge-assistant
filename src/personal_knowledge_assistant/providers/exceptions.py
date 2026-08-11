class ProviderError(Exception):
    """提供商调用失败。"""


class EmbeddingProviderError(ProviderError):
    """Embedding 提供商调用失败。"""


class EmptyEmbeddingInputError(EmbeddingProviderError):
    """Embedding 输入为空。"""


class EmbeddingDimensionError(EmbeddingProviderError):
    """Embedding 返回的向量维度不正确。"""
