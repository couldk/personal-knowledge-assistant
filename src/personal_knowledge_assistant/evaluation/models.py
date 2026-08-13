from enum import StrEnum
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EvaluationCategory(StrEnum):
    """评测问题类别。"""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    CROSS_DOCUMENT = "cross_document"
    MULTILINGUAL = "multilingual"
    OUT_OF_SCOPE = "out_of_scope"


class RetrievalEvaluationCase(BaseModel):
    """一条文件级检索黄金问题。"""

    id: str = Field(
        pattern=r"^q[0-9]{3}$",
    )
    question: str = Field(
        min_length=1,
    )
    expected_file_names: list[str] = Field(
        min_length=1,
    )
    category: EvaluationCategory
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
    )

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


class AnswerEvaluationCase(BaseModel):
    """一条完整回答评测问题。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    id: str = Field(
        pattern=r"^q[0-9]{3}$",
    )
    question: str = Field(
        min_length=1,
    )
    expected_file_names: list[str] = Field(
        default_factory=list,
    )
    expected_answer_terms: list[str] = Field(
        default_factory=list,
    )
    expected_refused: bool
    category: EvaluationCategory
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(
        cls,
        value: str,
    ) -> str:
        """清理问题并拒绝空白问题。"""

        normalized = value.strip()

        if not normalized:
            raise ValueError("Question cannot be blank.")

        return normalized

    @field_validator("expected_file_names")
    @classmethod
    def normalize_expected_file_names(
        cls,
        values: list[str],
    ) -> list[str]:
        """清理并校验期望文件名。"""

        normalized = [value.strip() for value in values]

        if any(not value for value in normalized):
            raise ValueError("Expected file names cannot be blank.")

        if len(normalized) != len(set(normalized)):
            raise ValueError("Expected file names must be unique.")

        return normalized

    @field_validator("expected_answer_terms")
    @classmethod
    def normalize_expected_answer_terms(
        cls,
        values: list[str],
    ) -> list[str]:
        """清理并校验期望答案关键词。"""

        normalized = [value.strip() for value in values]

        if any(not value for value in normalized):
            raise ValueError("Expected answer terms cannot be blank.")

        if len(normalized) != len(set(normalized)):
            raise ValueError("Expected answer terms must be unique.")

        return normalized

    @model_validator(mode="after")
    def validate_expected_answer_state(
        self,
    ) -> Self:
        """校验可回答和拒答问题的业务状态。"""

        if self.expected_refused:
            if self.category is not EvaluationCategory.OUT_OF_SCOPE:
                raise ValueError("Refusal cases must use the out_of_scope category.")

            if self.expected_file_names:
                raise ValueError("Refusal cases cannot have expected files.")

            if self.expected_answer_terms:
                raise ValueError("Refusal cases cannot have expected answer terms.")

            return self

        if self.category is EvaluationCategory.OUT_OF_SCOPE:
            raise ValueError("Out-of-scope cases must expect refusal.")

        if not self.expected_file_names:
            raise ValueError("Answerable cases require expected files.")

        if not self.expected_answer_terms:
            raise ValueError("Answerable cases require expected answer terms.")

        if self.category is EvaluationCategory.CROSS_DOCUMENT and len(self.expected_file_names) < 2:
            raise ValueError("Cross-document cases require at least two expected files.")

        return self


class RetrievalCaseResult(BaseModel):
    """一条检索问题的实际评测结果。"""

    case_id: str = Field(
        pattern=r"^q[0-9]{3}$",
    )
    category: EvaluationCategory
    expected_file_names: list[str] = Field(
        min_length=1,
    )
    retrieved_file_names: list[str] = Field(
        default_factory=list,
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
    )
    scores: list[float] = Field(
        default_factory=list,
    )
    first_relevant_rank: int | None = Field(
        default=None,
        ge=1,
    )
    hit_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )
    recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )
    reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )
    duration_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def result_lists_must_have_equal_lengths(
        self,
    ) -> Self:
        result_count = len(self.retrieved_file_names)

        if len(self.retrieved_chunk_ids) != result_count:
            raise ValueError("Retrieved file names and chunk IDs must align.")

        if len(self.scores) != result_count:
            raise ValueError("Retrieved file names and scores must align.")

        return self


class RetrievalEvaluationReport(BaseModel):
    """一次完整检索基线评测报告。"""

    generated_at: AwareDatetime
    model: str = Field(
        min_length=1,
    )
    dimension: int = Field(
        ge=1,
    )
    chunk_size: int = Field(
        ge=1,
    )
    chunk_overlap: int = Field(
        ge=0,
    )
    case_count: int = Field(
        ge=1,
    )
    hit_at_1: float = Field(
        ge=0.0,
        le=1.0,
    )
    recall_at_3: float = Field(
        ge=0.0,
        le=1.0,
    )
    mean_reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_duration_ms: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )
    cases: list[RetrievalCaseResult] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def case_count_must_match_results(
        self,
    ) -> Self:
        if self.case_count != len(self.cases):
            raise ValueError("Case count must match the number of results.")

        return self
