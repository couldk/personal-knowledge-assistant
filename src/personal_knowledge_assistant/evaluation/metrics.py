from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetricValues:
    """单个问题的文件级检索指标。"""

    first_relevant_rank: int | None
    hit_at_1: float
    recall_at_3: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class AnswerMetricValues:
    """单个回答问题的确定性质量指标。"""

    retrieval_recall_at_3: float
    citation_precision: float
    citation_recall: float
    answer_term_coverage: float
    refusal_correct: bool


def unique_in_order(
    values: Sequence[str],
) -> list[str]:
    """按照首次出现顺序对字符串去重。"""

    seen: set[str] = set()
    unique: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        unique.append(value)

    return unique


def calculate_retrieval_metrics(
    *,
    expected_file_names: Sequence[str],
    retrieved_file_names: Sequence[str],
    recall_k: int = 3,
) -> RetrievalMetricValues:
    """计算文件级 Hit@1、Recall@k 和 MRR。"""

    if not expected_file_names:
        raise ValueError("Expected file names cannot be empty.")

    if recall_k <= 0:
        raise ValueError("Recall k must be positive.")

    expected = set(expected_file_names)
    retrieved = unique_in_order(retrieved_file_names)

    first_relevant_rank = next(
        (
            rank
            for rank, file_name in enumerate(
                retrieved,
                start=1,
            )
            if file_name in expected
        ),
        None,
    )

    hit_at_1 = 1.0 if retrieved and retrieved[0] in expected else 0.0

    recall_at_k = len(expected & set(retrieved[:recall_k])) / len(expected)

    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0

    return RetrievalMetricValues(
        first_relevant_rank=first_relevant_rank,
        hit_at_1=hit_at_1,
        recall_at_3=recall_at_k,
        reciprocal_rank=reciprocal_rank,
    )


def calculate_answer_metrics(
    *,
    expected_file_names: Sequence[str],
    expected_answer_terms: Sequence[str],
    expected_refused: bool,
    retrieved_file_names: Sequence[str],
    cited_file_names: Sequence[str],
    answer: str,
    actual_refused: bool,
    recall_k: int = 3,
) -> AnswerMetricValues:
    """计算回答检索、引用、关键词和拒答指标。"""

    if recall_k <= 0:
        raise ValueError("Recall k must be positive.")

    if expected_refused:
        if expected_file_names:
            raise ValueError("Refusal cases cannot have expected files.")

        if expected_answer_terms:
            raise ValueError("Refusal cases cannot have expected answer terms.")
    else:
        if not expected_file_names:
            raise ValueError("Answerable cases require expected files.")

        if not expected_answer_terms:
            raise ValueError("Answerable cases require expected answer terms.")

    normalized_expected_files = {file_name.strip() for file_name in expected_file_names}

    normalized_retrieved_files = unique_in_order(
        [file_name.strip() for file_name in retrieved_file_names if file_name.strip()]
    )

    normalized_cited_files = [
        file_name.strip() for file_name in cited_file_names if file_name.strip()
    ]

    if normalized_expected_files:
        retrieval_recall_at_k = len(
            normalized_expected_files & set(normalized_retrieved_files[:recall_k])
        ) / len(normalized_expected_files)
    else:
        # 拒答问题没有黄金文件，该指标不适用。
        retrieval_recall_at_k = 1.0

    if normalized_cited_files:
        relevant_citation_count = sum(
            file_name in normalized_expected_files for file_name in normalized_cited_files
        )

        citation_precision = relevant_citation_count / len(normalized_cited_files)
    else:
        citation_precision = 1.0 if expected_refused and actual_refused else 0.0

    if normalized_expected_files:
        citation_recall = len(normalized_expected_files & set(normalized_cited_files)) / len(
            normalized_expected_files
        )
    else:
        citation_recall = 1.0 if expected_refused and actual_refused else 0.0

    normalized_answer = answer.casefold()

    normalized_terms = [term.strip().casefold() for term in expected_answer_terms if term.strip()]

    if normalized_terms:
        matched_term_count = sum(term in normalized_answer for term in normalized_terms)

        answer_term_coverage = matched_term_count / len(normalized_terms)
    else:
        answer_term_coverage = 1.0 if expected_refused and actual_refused else 0.0

    return AnswerMetricValues(
        retrieval_recall_at_3=(retrieval_recall_at_k),
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        answer_term_coverage=(answer_term_coverage),
        refusal_correct=(actual_refused == expected_refused),
    )
