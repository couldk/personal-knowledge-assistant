from enum import StrEnum
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class EvaluationCategory(StrEnum):
    """检索问题类别。"""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    CROSS_DOCUMENT = "cross_document"
    MULTILINGUAL = "multilingual"


class RetrievalEvaluationCase(BaseModel):
    """一条文件级检索黄金问题。"""

    id: str = Field(pattern=r"^q[0-9]{3}$")
    question: str = Field(min_length=1)
    expected_file_names: list[str] = Field(min_length=1)
    category: EvaluationCategory
    top_k: int = Field(default=3, ge=1, le=20)

    @field_validator("expected_file_names")
    @classmethod
    def expected_files_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [value.strip() for value in values]

        if any(not value for value in normalized):
            raise ValueError("Expected file names cannot be blank.")

        if len(set(normalized)) != len(normalized):
            raise ValueError("Expected file names must be unique.")

        return normalized


class RetrievalCaseResult(BaseModel):
    """一条检索问题的实际评测结果。"""

    case_id: str = Field(pattern=r"^q[0-9]{3}$")
    category: EvaluationCategory
    expected_file_names: list[str] = Field(min_length=1)
    retrieved_file_names: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    first_relevant_rank: int | None = Field(default=None, ge=1)
    hit_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    duration_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def result_lists_must_have_equal_lengths(self) -> Self:
        result_count = len(self.retrieved_file_names)

        if len(self.retrieved_chunk_ids) != result_count:
            raise ValueError("Retrieved file names and chunk IDs must align.")

        if len(self.scores) != result_count:
            raise ValueError("Retrieved file names and scores must align.")

        return self


class RetrievalEvaluationReport(BaseModel):
    """一次完整检索基线评测报告。"""

    generated_at: AwareDatetime
    model: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    case_count: int = Field(ge=1)
    hit_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_duration_ms: float = Field(ge=0.0, allow_inf_nan=False)
    cases: list[RetrievalCaseResult] = Field(min_length=1)

    @model_validator(mode="after")
    def case_count_must_match_results(self) -> Self:
        if self.case_count != len(self.cases):
            raise ValueError("Case count must match the number of results.")

        return self
