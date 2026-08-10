from typing import Protocol

from personal_knowledge_assistant.domain import LoadedDocument


class DocumentCatalog(Protocol):
    """保存和查询已经导入的文档。"""

    def get(
        self,
        document_id: str,
    ) -> LoadedDocument | None:
        """根据文档ID查询已经保存的文档。"""

        ...

    def save(
        self,
        document: LoadedDocument,
    ) -> None:
        """创建或覆盖一个文档。"""

        ...


class InMemoryDocumentCatalog:
    """仅在当前Python进程中保存文档。"""

    def __init__(self) -> None:
        self._documents: dict[str, LoadedDocument] = {}

    def get(
        self,
        document_id: str,
    ) -> LoadedDocument | None:
        """查询文档，并返回独立副本。"""

        document = self._documents.get(document_id)

        if document is None:
            return None

        return document.model_copy(deep=True)

    def save(
        self,
        document: LoadedDocument,
    ) -> None:
        """保存文档的独立副本，避免外部修改内部数据。"""

        self._documents[document.document_id] = document.model_copy(
            deep=True,
        )

    def __len__(self) -> int:
        """返回当前保存的文档数量。"""

        return len(self._documents)
