class ChunkingError(Exception):
    """所有文档切块异常的基类。"""


class EmptyChunkingResultError(ChunkingError):
    """文档没有产生任何可索引Chunk。"""

    def __init__(
        self,
        document_id: str,
    ) -> None:
        self.document_id = document_id

        super().__init__(f"Document produced no searchable chunks: {document_id}")
