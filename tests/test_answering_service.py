import json
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from personal_knowledge_assistant.answering import (
    AnsweringService,
    EvidencePromptBuilder,
    InvalidAnswerResponseError,
    UnknownCitationError,
    create_answering_service,
)
from personal_knowledge_assistant.config import Settings
from personal_knowledge_assistant.domain.documents import (
    DocumentType,
)
from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from personal_knowledge_assistant.domain.retrieval import (
    ChunkMetadata,
    DocumentChunk,
    SearchResult,
)
from personal_knowledge_assistant.providers.exceptions import (
    ChatProviderError,
)


class FakeChatProvider:
    """用于 AnsweringService 单元测试的聊天 Provider。"""

    def __init__(
        self,
        *,
        response_content: str = "",
        error: Exception | None = None,
    ) -> None:
        self.response_content = response_content
        self.error = error
        self.messages: list[ChatMessage] = []
        self.json_mode = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool = False,
    ) -> ChatResponse:
        self.messages = list(messages)
        self.json_mode = json_mode

        if self.error is not None:
            raise self.error

        return ChatResponse(
            content=self.response_content,
            model="test-chat-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )


def _make_result(
    *,
    chunk_character: str = "a",
    text: str = "RAG combines retrieval with generation.",
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_character * 64,
            document_id="document-1",
            text=text,
            metadata=ChunkMetadata(
                content_hash="f" * 64,
                source_path="D:/private/knowledge.md",
                file_name="knowledge.md",
                document_type=DocumentType.MARKDOWN,
                modified_at=datetime(
                    2026,
                    1,
                    1,
                    tzinfo=UTC,
                ),
                part_index=0,
                chunk_index=0,
                page_number=1,
                section_path=[
                    "Agent",
                    "RAG",
                ],
            ),
        ),
        score=0.95,
    )


def _answer_json(
    *,
    answer: str = "RAG combines retrieval and generation.",
    citation_ids: Sequence[str] = ("a" * 64,),
    confidence: float = 0.9,
    refused: bool = False,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": [
                {
                    "chunk_id": chunk_id,
                }
                for chunk_id in citation_ids
            ],
            "confidence": confidence,
            "refused": refused,
        }
    )


def _make_service(
    *,
    response_content: str,
    error: Exception | None = None,
) -> tuple[AnsweringService, FakeChatProvider]:
    provider = FakeChatProvider(
        response_content=response_content,
        error=error,
    )

    service = AnsweringService(
        chat_provider=provider,
        prompt_builder=EvidencePromptBuilder(),
    )

    return service, provider


@pytest.mark.asyncio
async def test_answer_returns_valid_evidence_answer() -> None:
    service, provider = _make_service(response_content=_answer_json())

    answer = await service.answer(
        question="什么是 RAG？",
        results=[_make_result()],
    )

    assert answer.answer == ("RAG combines retrieval and generation.")
    assert answer.confidence == 0.9
    assert answer.refused is False
    assert [citation.chunk_id for citation in answer.citations] == ["a" * 64]

    assert provider.json_mode is True
    assert len(provider.messages) == 2
    assert provider.messages[0].role is MessageRole.SYSTEM
    assert provider.messages[1].role is MessageRole.USER


@pytest.mark.asyncio
async def test_answer_passes_question_and_evidence_to_prompt() -> None:
    service, provider = _make_service(response_content=_answer_json())

    await service.answer(
        question="什么是 RAG？",
        results=[_make_result()],
    )

    user_message = provider.messages[1].content

    assert "什么是 RAG？" in user_message
    assert "a" * 64 in user_message
    assert "RAG combines retrieval with generation." in user_message


@pytest.mark.asyncio
async def test_answer_accepts_known_multiple_citations() -> None:
    service, _ = _make_service(
        response_content=_answer_json(
            citation_ids=[
                "a" * 64,
                "b" * 64,
            ]
        )
    )

    answer = await service.answer(
        question="比较两个证据。",
        results=[
            _make_result(
                chunk_character="a",
                text="First evidence.",
            ),
            _make_result(
                chunk_character="b",
                text="Second evidence.",
            ),
        ],
    )

    assert [citation.chunk_id for citation in answer.citations] == [
        "a" * 64,
        "b" * 64,
    ]


@pytest.mark.asyncio
async def test_answer_accepts_refusal_without_evidence() -> None:
    service, _ = _make_service(
        response_content=_answer_json(
            answer="现有证据不足，无法回答。",
            citation_ids=[],
            confidence=0.0,
            refused=True,
        )
    )

    answer = await service.answer(
        question="一个没有相关资料的问题。",
        results=[],
    )

    assert answer.refused is True
    assert answer.citations == []
    assert answer.confidence == 0.0


@pytest.mark.asyncio
async def test_answer_rejects_invalid_json() -> None:
    service, _ = _make_service(response_content="not valid JSON")

    with pytest.raises(
        InvalidAnswerResponseError,
        match="invalid EvidenceAnswer",
    ):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )


@pytest.mark.asyncio
async def test_answer_rejects_invalid_schema() -> None:
    service, _ = _make_service(
        response_content=json.dumps(
            {
                "answer": "Invalid response",
                "citations": [],
                "confidence": 2.0,
                "refused": False,
            }
        )
    )

    with pytest.raises(
        InvalidAnswerResponseError,
        match="invalid EvidenceAnswer",
    ):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )


@pytest.mark.asyncio
async def test_answer_rejects_unknown_citation() -> None:
    service, _ = _make_service(response_content=_answer_json(citation_ids=["b" * 64]))

    with pytest.raises(
        UnknownCitationError,
    ) as exc_info:
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )

    assert exc_info.value.chunk_ids == ("b" * 64,)


@pytest.mark.asyncio
async def test_answer_rejects_partially_unknown_citations() -> None:
    service, _ = _make_service(
        response_content=_answer_json(
            citation_ids=[
                "a" * 64,
                "c" * 64,
            ]
        )
    )

    with pytest.raises(
        UnknownCitationError,
    ) as exc_info:
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result(chunk_character="a")],
        )

    assert exc_info.value.chunk_ids == ("c" * 64,)


@pytest.mark.asyncio
async def test_answer_rejects_non_refused_answer_without_citations() -> None:
    service, _ = _make_service(
        response_content=_answer_json(
            citation_ids=[],
            refused=False,
        )
    )

    with pytest.raises(
        InvalidAnswerResponseError,
    ):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )


@pytest.mark.asyncio
async def test_answer_propagates_provider_error() -> None:
    service, _ = _make_service(
        response_content="",
        error=ChatProviderError("Provider unavailable."),
    )

    with pytest.raises(
        ChatProviderError,
        match="Provider unavailable",
    ):
        await service.answer(
            question="什么是 RAG？",
            results=[_make_result()],
        )


@pytest.mark.asyncio
async def test_answer_rejects_empty_question() -> None:
    service, provider = _make_service(response_content=_answer_json())

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        await service.answer(
            question="   ",
            results=[_make_result()],
        )

    assert provider.messages == []


def test_factory_uses_injected_chat_provider() -> None:
    provider = FakeChatProvider(response_content=_answer_json())

    settings = Settings.model_validate(
        {
            "chat_provider": "deepseek",
            "chat_model": "test-model",
            "chat_api_key": "test-key",
            "chat_base_url": "https://api.deepseek.com",
        }
    )

    service = create_answering_service(
        settings=settings,
        chat_provider=provider,
    )

    assert isinstance(service, AnsweringService)
