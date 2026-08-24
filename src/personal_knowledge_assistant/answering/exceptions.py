from collections.abc import Sequence


class AnsweringError(Exception):
    """回答生成流程基础异常。"""


class InvalidAnswerResponseError(AnsweringError):
    """聊天模型返回的内容不符合回答 Schema。"""


class UnknownCitationError(AnsweringError):
    """回答引用了本次检索结果中不存在的 Chunk。"""

    def __init__(
        self,
        chunk_ids: Sequence[str],
    ) -> None:
        self.chunk_ids = tuple(chunk_ids)

        joined_chunk_ids = ", ".join(self.chunk_ids)

        super().__init__(
            f"Answer cites chunks not present in retrieval results: {joined_chunk_ids}"
        )
