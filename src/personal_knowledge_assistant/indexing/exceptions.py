class IndexingError(Exception):
    """文档索引基础异常。"""


class EmbeddingCountMismatchError(IndexingError):
    """Chunk 数量和 Embedding 数量不一致。"""
