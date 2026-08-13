from collections.abc import Sequence

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
    ) -> None:
        self._chat_provider = chat_provider
        self._prompt_builder = prompt_builder

    async def answer(
        self,
        *,
        question: str,
        results: Sequence[SearchResult],
    ) -> EvidenceAnswer:
        """生成基于检索证据的结构化回答。"""

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
