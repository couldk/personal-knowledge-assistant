class VectorStoreError(Exception):
    """向量存储基础异常。"""


class VectorCountMismatchError(VectorStoreError):
    """Chunk 数量和向量数量不一致。"""


class VectorDimensionMismatchError(VectorStoreError):
    """向量维度与向量库配置不一致。"""


class InvalidVectorError(VectorStoreError):
    """向量包含非法数值或为全零向量。"""


class UnsupportedVectorFilterError(VectorStoreError):
    """使用了不支持的向量检索过滤字段。"""
