from personal_knowledge_assistant.chunking.base import (
    DocumentChunker,
)
from personal_knowledge_assistant.domain import (
    ImportResult,
    ImportStatus,
)
from personal_knowledge_assistant.indexing.exceptions import (
    EmbeddingCountMismatchError,
)
from personal_knowledge_assistant.indexing.models import (
    IndexingResult,
    IndexingStatus,
)
from personal_knowledge_assistant.providers.base import (
    EmbeddingProvider,
    VectorStoreProvider,
)


class DocumentIndexingService:
    """把导入完成的文档转换成可检索向量。"""

    def __init__(
        self,
        *,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
    ) -> None:
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def index(
        self,
        import_result: ImportResult,
    ) -> IndexingResult:
        """切块、生成向量并写入向量库。"""

        document = import_result.document

        if import_result.status == ImportStatus.UNCHANGED:
            return IndexingResult(
                status=IndexingStatus.SKIPPED,
                document_id=document.document_id,
                content_hash=document.content_hash,
                chunk_count=0,
                deactivated_chunk_count=0,
            )

        chunks = self._chunker.split(document)

        texts = [chunk.text for chunk in chunks]

        # 必须先成功生成全部新向量，再停用旧版本。
        embeddings = await self._embedding_provider.embed_texts(texts)

        if len(embeddings) != len(chunks):
            raise EmbeddingCountMismatchError(
                f"Expected {len(chunks)} embeddings, received {len(embeddings)}."
            )

        deactivated_chunk_count = 0

        if import_result.status == ImportStatus.UPDATED:
            deactivated_chunk_count = await self._vector_store.deactivate_document(
                document.document_id
            )

        await self._vector_store.upsert(
            chunks,
            embeddings,
        )

        return IndexingResult(
            status=IndexingStatus.INDEXED,
            document_id=document.document_id,
            content_hash=document.content_hash,
            chunk_count=len(chunks),
            deactivated_chunk_count=(deactivated_chunk_count),
        )
