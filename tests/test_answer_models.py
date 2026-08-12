import pytest
from pydantic import ValidationError

from personal_knowledge_assistant.answering import (
    AnswerCitation,
    EvidenceAnswer,
)

VALID_CHUNK_ID = "a" * 64
SECOND_CHUNK_ID = "b" * 64


def test_answer_citation_accepts_valid_chunk_id() -> None:
    citation = AnswerCitation(
        chunk_id=VALID_CHUNK_ID,
    )

    assert citation.chunk_id == VALID_CHUNK_ID


@pytest.mark.parametrize(
    "invalid_chunk_id",
    [
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "not-a-chunk-id",
        "g" * 64,
    ],
)
def test_answer_citation_rejects_invalid_chunk_id(
    invalid_chunk_id: str,
) -> None:
    with pytest.raises(ValidationError):
        AnswerCitation(
            chunk_id=invalid_chunk_id,
        )


def test_non_refused_answer_requires_citation() -> None:
    answer = EvidenceAnswer(
        answer=("向量检索通过比较查询向量和文档向量的相似度寻找相关内容。"),
        citations=[
            AnswerCitation(
                chunk_id=VALID_CHUNK_ID,
            )
        ],
        confidence=0.9,
        refused=False,
    )

    assert answer.refused is False
    assert answer.confidence == 0.9
    assert answer.citations == [
        AnswerCitation(
            chunk_id=VALID_CHUNK_ID,
        )
    ]


def test_non_refused_answer_accepts_multiple_citations() -> None:
    answer = EvidenceAnswer(
        answer="该结论由两个检索片段共同支持。",
        citations=[
            AnswerCitation(
                chunk_id=VALID_CHUNK_ID,
            ),
            AnswerCitation(
                chunk_id=SECOND_CHUNK_ID,
            ),
        ],
        confidence=0.8,
        refused=False,
    )

    assert len(answer.citations) == 2


def test_non_refused_answer_rejects_empty_citations() -> None:
    with pytest.raises(
        ValidationError,
        match=("Non-refused answers require at least one citation"),
    ):
        EvidenceAnswer(
            answer="没有引用的正常回答。",
            citations=[],
            confidence=0.5,
            refused=False,
        )


def test_refused_answer_accepts_empty_citations() -> None:
    answer = EvidenceAnswer(
        answer=("现有知识库中没有足够证据回答这个问题。"),
        citations=[],
        confidence=0.0,
        refused=True,
    )

    assert answer.refused is True
    assert answer.citations == []
    assert answer.confidence == 0.0


def test_refused_answer_rejects_citations() -> None:
    with pytest.raises(
        ValidationError,
        match=("Refused answers cannot contain citations"),
    ):
        EvidenceAnswer(
            answer="证据不足。",
            citations=[
                AnswerCitation(
                    chunk_id=VALID_CHUNK_ID,
                )
            ],
            confidence=0.0,
            refused=True,
        )


def test_answer_rejects_duplicate_citations() -> None:
    with pytest.raises(
        ValidationError,
        match="Answer citations must be unique",
    ):
        EvidenceAnswer(
            answer="回答正文。",
            citations=[
                AnswerCitation(
                    chunk_id=VALID_CHUNK_ID,
                ),
                AnswerCitation(
                    chunk_id=VALID_CHUNK_ID,
                ),
            ],
            confidence=0.8,
            refused=False,
        )


@pytest.mark.parametrize(
    "invalid_confidence",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_answer_rejects_invalid_confidence(
    invalid_confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceAnswer(
            answer="回答正文。",
            citations=[
                AnswerCitation(
                    chunk_id=VALID_CHUNK_ID,
                )
            ],
            confidence=invalid_confidence,
            refused=False,
        )


@pytest.mark.parametrize(
    "valid_confidence",
    [
        0.0,
        0.5,
        1.0,
    ],
)
def test_answer_accepts_confidence_boundaries(
    valid_confidence: float,
) -> None:
    answer = EvidenceAnswer(
        answer="回答正文。",
        citations=[
            AnswerCitation(
                chunk_id=VALID_CHUNK_ID,
            )
        ],
        confidence=valid_confidence,
        refused=False,
    )

    assert answer.confidence == valid_confidence


@pytest.mark.parametrize(
    "invalid_answer",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
        "\n\t",
    ],
)
def test_answer_rejects_empty_text(
    invalid_answer: str,
) -> None:
    with pytest.raises(ValidationError):
        EvidenceAnswer(
            answer=invalid_answer,
            citations=[],
            confidence=0.0,
            refused=True,
        )


def test_answer_can_be_parsed_from_model_json() -> None:
    payload = """
    {
        "answer": "更换模型后必须重建向量索引。",
        "citations": [
            {
                "chunk_id":
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }
        ],
        "confidence": 0.95,
        "refused": false
    }
    """

    answer = EvidenceAnswer.model_validate_json(payload)

    assert answer.answer == ("更换模型后必须重建向量索引。")
    assert answer.citations[0].chunk_id == (VALID_CHUNK_ID)
    assert answer.confidence == 0.95
    assert answer.refused is False


def test_answer_serializes_to_expected_shape() -> None:
    answer = EvidenceAnswer(
        answer="证据不足。",
        citations=[],
        confidence=0.0,
        refused=True,
    )

    assert answer.model_dump() == {
        "answer": "证据不足。",
        "citations": [],
        "confidence": 0.0,
        "refused": True,
    }


def test_answer_strips_surrounding_whitespace() -> None:
    answer = EvidenceAnswer(
        answer="  现有证据不足。  ",
        citations=[],
        confidence=0.0,
        refused=True,
    )

    assert answer.answer == "现有证据不足。"
