from typing import Annotated, Self

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

from personal_knowledge_assistant.domain.retrieval import (
    ChunkId,
)

NonEmptyAnswer = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]


class AnswerCitation(BaseModel):
    """结构化回答引用的检索片段。"""

    chunk_id: ChunkId


class EvidenceAnswer(BaseModel):
    """基于检索证据生成的结构化回答。"""

    answer: NonEmptyAnswer

    citations: list[AnswerCitation] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )

    refused: bool

    @model_validator(mode="after")
    def validate_answer_state(self) -> Self:
        """确保回答、拒答和引用状态一致。"""

        citation_ids = [citation.chunk_id for citation in self.citations]

        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Answer citations must be unique.")

        if self.refused and self.citations:
            raise ValueError("Refused answers cannot contain citations.")

        if not self.refused and not self.citations:
            raise ValueError("Non-refused answers require at least one citation.")

        return self
