from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetricValues:
    """单个问题的文件级检索指标。"""

    first_relevant_rank: int | None
    hit_at_1: float
    recall_at_3: float
    reciprocal_rank: float


def unique_in_order(values: Sequence[str]) -> list[str]:
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
        (rank for rank, file_name in enumerate(retrieved, start=1) if file_name in expected),
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
