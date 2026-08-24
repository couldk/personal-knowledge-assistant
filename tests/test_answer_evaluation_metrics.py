import pytest

from personal_knowledge_assistant.evaluation import (
    AnswerMetricValues,
    calculate_answer_metrics,
)


def test_perfect_answer_metrics() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "检索",
            "生成",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
            "security.md",
        ],
        cited_file_names=[
            "rag.md",
        ],
        answer=("RAG 首先检索相关证据，然后生成回答。"),
        actual_refused=False,
    )

    assert metrics == AnswerMetricValues(
        retrieval_recall_at_3=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        answer_term_coverage=1.0,
        refusal_correct=True,
    )


def test_partial_multi_document_metrics() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
            "security.md",
        ],
        expected_answer_terms=[
            "检索",
            "后置校验",
            "不可信数据",
            "Chunk ID",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
            "memory.md",
        ],
        cited_file_names=[
            "rag.md",
        ],
        answer=("系统使用检索结果和 Chunk ID。"),
        actual_refused=False,
    )

    assert metrics.retrieval_recall_at_3 == 0.5
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 0.5
    assert metrics.answer_term_coverage == 0.5
    assert metrics.refusal_correct is True


def test_incorrect_citation_reduces_precision() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "检索",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
            "security.md",
        ],
        cited_file_names=[
            "rag.md",
            "security.md",
        ],
        answer="系统先进行检索。",
        actual_refused=False,
    )

    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 1.0


def test_duplicate_citations_count_in_precision() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "检索",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[
            "rag.md",
            "rag.md",
            "security.md",
        ],
        answer="系统先进行检索。",
        actual_refused=False,
    )

    assert metrics.citation_precision == pytest.approx(2 / 3)
    assert metrics.citation_recall == 1.0


def test_missing_citations_score_zero() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "检索",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[],
        answer="系统先进行检索。",
        actual_refused=False,
    )

    assert metrics.citation_precision == 0.0
    assert metrics.citation_recall == 0.0


def test_correct_refusal_receives_full_applicable_scores() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[],
        expected_answer_terms=[],
        expected_refused=True,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[],
        answer="现有证据不足，无法回答。",
        actual_refused=True,
    )

    assert metrics.retrieval_recall_at_3 == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0
    assert metrics.answer_term_coverage == 1.0
    assert metrics.refusal_correct is True


def test_incorrect_non_refusal_fails_refusal_metrics() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[],
        expected_answer_terms=[],
        expected_refused=True,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[],
        answer="我猜答案是 42。",
        actual_refused=False,
    )

    assert metrics.retrieval_recall_at_3 == 1.0
    assert metrics.citation_precision == 0.0
    assert metrics.citation_recall == 0.0
    assert metrics.answer_term_coverage == 0.0
    assert metrics.refusal_correct is False


def test_term_matching_is_case_insensitive() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "RAG",
            "Chunk ID",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[
            "rag.md",
        ],
        answer=("rag 使用检索结果，并验证 CHUNK ID。"),
        actual_refused=False,
    )

    assert metrics.answer_term_coverage == 1.0


def test_term_matching_supports_chinese() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "memory.md",
        ],
        expected_answer_terms=[
            "长期记忆",
            "过期策略",
            "置信度",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "memory.md",
        ],
        cited_file_names=[
            "memory.md",
        ],
        answer=("长期记忆需要置信度信息，但这里没有讨论保留期限。"),
        actual_refused=False,
    )

    assert metrics.answer_term_coverage == pytest.approx(2 / 3)


def test_retrieval_recall_uses_unique_top_k_files() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
            "security.md",
        ],
        expected_answer_terms=[
            "检索",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
            "rag.md",
            "memory.md",
            "security.md",
        ],
        cited_file_names=[
            "rag.md",
            "security.md",
        ],
        answer="系统使用检索证据。",
        actual_refused=False,
        recall_k=3,
    )

    assert metrics.retrieval_recall_at_3 == 1.0


def test_wrong_refusal_is_detected() -> None:
    metrics = calculate_answer_metrics(
        expected_file_names=[
            "rag.md",
        ],
        expected_answer_terms=[
            "检索",
        ],
        expected_refused=False,
        retrieved_file_names=[
            "rag.md",
        ],
        cited_file_names=[],
        answer="无法回答。",
        actual_refused=True,
    )

    assert metrics.refusal_correct is False
    assert metrics.citation_precision == 0.0
    assert metrics.citation_recall == 0.0


def test_recall_k_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="Recall k must be positive",
    ):
        calculate_answer_metrics(
            expected_file_names=[
                "rag.md",
            ],
            expected_answer_terms=[
                "检索",
            ],
            expected_refused=False,
            retrieved_file_names=[
                "rag.md",
            ],
            cited_file_names=[
                "rag.md",
            ],
            answer="检索",
            actual_refused=False,
            recall_k=0,
        )


def test_refusal_case_rejects_expected_files() -> None:
    with pytest.raises(
        ValueError,
        match=("Refusal cases cannot have expected files"),
    ):
        calculate_answer_metrics(
            expected_file_names=[
                "rag.md",
            ],
            expected_answer_terms=[],
            expected_refused=True,
            retrieved_file_names=[],
            cited_file_names=[],
            answer="无法回答。",
            actual_refused=True,
        )


def test_answerable_case_requires_expected_terms() -> None:
    with pytest.raises(
        ValueError,
        match=("Answerable cases require expected answer terms"),
    ):
        calculate_answer_metrics(
            expected_file_names=[
                "rag.md",
            ],
            expected_answer_terms=[],
            expected_refused=False,
            retrieved_file_names=[
                "rag.md",
            ],
            cited_file_names=[
                "rag.md",
            ],
            answer="检索",
            actual_refused=False,
        )
