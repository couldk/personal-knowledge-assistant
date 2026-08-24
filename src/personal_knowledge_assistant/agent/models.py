from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from personal_knowledge_assistant.agent.state import (
    AgentState,
    NonEmptyText,
)
from personal_knowledge_assistant.answering import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain.retrieval import (
    ChunkId,
)


class AgentOutcome(StrEnum):
    """知识Agent最终执行结果。"""

    ANSWERED = "answered"
    REFUSED = "refused"
    FAILED = "failed"


class AgentEvidence(BaseModel):
    """Agent对外返回的证据摘要。

    不包含Chunk正文，避免把完整知识库内容直接暴露给调用方。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    chunk_id: ChunkId
    file_name: NonEmptyText

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    section_path: list[NonEmptyText] = Field(
        default_factory=list,
    )

    score: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )


class KnowledgeAgentResult(BaseModel):
    """一次知识Agent运行对外公开的结构化结果。"""

    model_config = ConfigDict(
        extra="forbid",
    )

    outcome: AgentOutcome
    answer: EvidenceAnswer

    retrieval_attempts: int = Field(
        ge=0,
        le=5,
    )

    final_retrieval_query: NonEmptyText

    evidence: list[AgentEvidence] = Field(
        default_factory=list,
    )

    failure_reason: NonEmptyText | None = None
    error_type: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> Self:
        """确保执行结果、拒答、异常和引用保持一致。"""

        evidence_ids = [item.chunk_id for item in self.evidence]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Agent evidence chunk IDs must be unique.")

        citation_ids = [citation.chunk_id for citation in self.answer.citations]

        unknown_citation_ids = set(citation_ids) - set(evidence_ids)

        if unknown_citation_ids:
            names = ", ".join(sorted(unknown_citation_ids))

            raise ValueError(f"Answer citations must reference returned agent evidence: {names}")

        if self.outcome is AgentOutcome.ANSWERED:
            if self.answer.refused:
                raise ValueError("An answered result cannot contain a refused answer.")

            if self.error_type is not None:
                raise ValueError("An answered result cannot contain an error type.")

        elif self.outcome is AgentOutcome.REFUSED:
            if not self.answer.refused:
                raise ValueError("A refused result requires a refused answer.")

            if self.error_type is not None:
                raise ValueError("A normal refusal cannot contain an error type.")

        else:
            if not self.answer.refused:
                raise ValueError("A failed result must return a safe refused answer.")

            if self.error_type is None:
                raise ValueError("A failed result requires an error type.")

            if self.failure_reason is None:
                raise ValueError("A failed result requires a failure reason.")

        return self


def create_agent_result(
    state: AgentState,
) -> KnowledgeAgentResult:
    """把LangGraph内部状态转换为安全的公开结果。"""

    answer = state.get("answer")

    if answer is None:
        raise ValueError("Agent state does not contain a final answer.")

    final_retrieval_query = state.get(
        "retrieval_query",
    )

    if final_retrieval_query is None:
        raise ValueError("Agent state does not contain a retrieval query.")

    retrieval_attempts = state.get(
        "retrieval_attempts",
    )

    if retrieval_attempts is None:
        raise ValueError("Agent state does not contain retrieval_attempts.")

    search_results = state.get(
        "retrieval_results",
        [],
    )

    evidence = [
        AgentEvidence(
            chunk_id=result.chunk.chunk_id,
            file_name=result.chunk.metadata.file_name,
            page_number=(result.chunk.metadata.page_number),
            section_path=(result.chunk.metadata.section_path.copy()),
            score=result.score,
        )
        for result in search_results
    ]

    error_type = state.get("error_type")
    failure_reason = state.get(
        "failure_reason",
    )

    if error_type is not None:
        outcome = AgentOutcome.FAILED
    elif answer.refused:
        outcome = AgentOutcome.REFUSED
    else:
        outcome = AgentOutcome.ANSWERED

    return KnowledgeAgentResult(
        outcome=outcome,
        answer=answer,
        retrieval_attempts=retrieval_attempts,
        final_retrieval_query=(final_retrieval_query),
        evidence=evidence,
        failure_reason=failure_reason,
        error_type=error_type,
    )
