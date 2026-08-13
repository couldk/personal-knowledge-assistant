from collections.abc import Sequence
from hashlib import sha256
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from personal_knowledge_assistant.answering.exceptions import (
    InvalidAnswerResponseError,
    UnknownCitationError,
)
from personal_knowledge_assistant.answering.models import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.answering.prompt_builder import (
    EvidencePromptBuilder,
)
from personal_knowledge_assistant.answering.trace_models import (
    AnsweringTrace,
    AnsweringTraceStatus,
)
from personal_knowledge_assistant.answering.tracing import (
    AnsweringTracer,
    LoggingAnsweringTracer,
)
from personal_knowledge_assistant.domain.models import (
    ChatResponse,
)
from personal_knowledge_assistant.domain.retrieval import (
    SearchResult,
)
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
)


class AnsweringService:
    """根据检索证据生成并验证结构化回答。"""

    def __init__(
        self,
        *,
        chat_provider: ChatProvider,
        prompt_builder: EvidencePromptBuilder,
        tracer: AnsweringTracer | None = None,
    ) -> None:
        self._chat_provider = chat_provider
        self._prompt_builder = prompt_builder
        self._tracer = tracer if tracer is not None else LoggingAnsweringTracer()

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        """生成基于检索证据的结构化回答。"""

        trace_id = uuid4()
        started_at = perf_counter()

        normalized_question = question.strip()
        question_hash = sha256(normalized_question.encode("utf-8")).hexdigest()

        response: ChatResponse | None = None

        try:
            messages = self._prompt_builder.build(
                question=question,
                results=results,
            )

            response = await self._chat_provider.complete(
                messages,
                json_mode=True,
            )

            try:
                answer = EvidenceAnswer.model_validate_json(response.content)
            except ValidationError as exc:
                raise InvalidAnswerResponseError(
                    "Chat provider returned an invalid EvidenceAnswer response."
                ) from exc

            self._validate_citations(
                answer=answer,
                results=results,
            )

        except Exception as exc:
            self._tracer.record(
                AnsweringTrace(
                    trace_id=trace_id,
                    status=AnsweringTraceStatus.FAILED,
                    question_hash=question_hash,
                    duration_ms=(perf_counter() - started_at) * 1000,
                    model=(response.model if response is not None else None),
                    refused=None,
                    citation_ids=[],
                    prompt_tokens=self._usage_value(
                        response,
                        "prompt_tokens",
                    ),
                    completion_tokens=self._usage_value(
                        response,
                        "completion_tokens",
                    ),
                    total_tokens=self._usage_value(
                        response,
                        "total_tokens",
                    ),
                    error_type=type(exc).__name__,
                )
            )

            raise

        # 成功执行到这里时，模型响应一定已经存在。
        assert response is not None

        self._tracer.record(
            AnsweringTrace(
                trace_id=trace_id,
                status=AnsweringTraceStatus.SUCCEEDED,
                question_hash=question_hash,
                duration_ms=(perf_counter() - started_at) * 1000,
                model=response.model,
                refused=answer.refused,
                citation_ids=[citation.chunk_id for citation in answer.citations],
                prompt_tokens=self._usage_value(
                    response,
                    "prompt_tokens",
                ),
                completion_tokens=self._usage_value(
                    response,
                    "completion_tokens",
                ),
                total_tokens=self._usage_value(
                    response,
                    "total_tokens",
                ),
                error_type=None,
            )
        )

        return answer

    @staticmethod
    def _validate_citations(
        *,
        answer: EvidenceAnswer,
        results: Sequence[SearchResult],
    ) -> None:
        """确保所有引用都来自当前检索结果。"""

        allowed_chunk_ids = {result.chunk.chunk_id for result in results}

        unknown_chunk_ids = [
            citation.chunk_id
            for citation in answer.citations
            if citation.chunk_id not in allowed_chunk_ids
        ]

        if unknown_chunk_ids:
            raise UnknownCitationError(unknown_chunk_ids)

    @staticmethod
    def _usage_value(
        response: ChatResponse | None,
        name: str,
    ) -> int:
        """安全读取模型 Token 用量。"""

        if response is None:
            return 0

        return response.usage.get(name, 0)
