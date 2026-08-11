from hashlib import sha256
from time import perf_counter
from uuid import uuid4

from personal_knowledge_assistant.domain import (
    RetrievalQuery,
    SearchResult,
)
from personal_knowledge_assistant.providers.base import (
    EmbeddingProvider,
    VectorStoreProvider,
)
from personal_knowledge_assistant.retrieval.models import (
    RetrievalHitTrace,
    RetrievalTrace,
    RetrievalTraceStatus,
)
from personal_knowledge_assistant.retrieval.tracing import (
    RetrievalTracer,
)


class RetrievalService:
    """根据自然语言问题检索相关文档片段。"""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        tracer: RetrievalTracer,
        embedding_model: str,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._tracer = tracer
        self._embedding_model = embedding_model

    async def retrieve(
        self,
        request: RetrievalQuery,
    ) -> list[SearchResult]:
        """生成查询向量并执行 top-k 搜索。"""

        trace_id = uuid4()
        started_at = perf_counter()

        query_hash = sha256(request.query.encode("utf-8")).hexdigest()

        try:
            query_embedding = await self._embedding_provider.embed_query(request.query)

            results = await self._vector_store.search(
                query_embedding,
                limit=request.top_k,
                filters=request.filters,
            )

        except Exception as exc:
            duration_ms = (perf_counter() - started_at) * 1000

            self._tracer.record(
                RetrievalTrace(
                    trace_id=trace_id,
                    status=(RetrievalTraceStatus.FAILED),
                    query_hash=query_hash,
                    top_k=request.top_k,
                    result_count=0,
                    duration_ms=duration_ms,
                    embedding_model=(self._embedding_model),
                    hits=[],
                    error_type=type(exc).__name__,
                )
            )

            raise

        duration_ms = (perf_counter() - started_at) * 1000

        self._tracer.record(
            RetrievalTrace(
                trace_id=trace_id,
                status=(RetrievalTraceStatus.SUCCEEDED),
                query_hash=query_hash,
                top_k=request.top_k,
                result_count=len(results),
                duration_ms=duration_ms,
                embedding_model=self._embedding_model,
                hits=[
                    RetrievalHitTrace(
                        chunk_id=(result.chunk.chunk_id),
                        score=result.score,
                    )
                    for result in results
                ],
                error_type=None,
            )
        )

        return results
