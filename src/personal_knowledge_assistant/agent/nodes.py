from personal_knowledge_assistant.agent.state import (
    AgentRoute,
    AgentState,
)
from personal_knowledge_assistant.answering import (
    EvidenceAnswer,
)
from personal_knowledge_assistant.domain import (
    ChatMessage,
    MessageRole,
    RetrievalQuery,
)
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
)
from personal_knowledge_assistant.querying.base import (
    AnswerGenerator,
    Retriever,
)

_REWRITE_SYSTEM_PROMPT = """
You rewrite search queries for a personal knowledge base.

Rules:
1. Preserve the user's original intent.
2. Make the query clearer and more suitable for semantic retrieval.
3. Do not answer the question.
4. Do not add facts that are not present in the original question.
5. Return only the rewritten query.
6. Do not wrap the query in quotes or Markdown.
""".strip()


class KnowledgeAgentNodes:
    """知识Agent的LangGraph节点集合。"""

    def __init__(
        self,
        *,
        retriever: Retriever,
        answer_generator: AnswerGenerator,
        chat_provider: ChatProvider,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._chat_provider = chat_provider

    async def retrieve(
        self,
        state: AgentState,
    ) -> AgentState:
        """使用当前查询检索知识库。"""

        retrieval_attempts = (
            state.get(
                "retrieval_attempts",
                0,
            )
            + 1
        )

        try:
            request = RetrievalQuery(
                query=state["retrieval_query"],
                top_k=state["top_k"],
                filters=state.get(
                    "filters",
                    {},
                ),
            )

            results = await self._retriever.retrieve(
                request,
            )

        except Exception as exc:
            return self._failure_update(
                failure_reason=("Knowledge retrieval failed."),
                error_type=type(exc).__name__,
                retrieval_attempts=(retrieval_attempts),
            )

        return AgentState(
            retrieval_attempts=retrieval_attempts,
            retrieval_results=results,
            route=AgentRoute.GRADE_EVIDENCE,
            failure_reason=None,
            error_type=None,
        )

    def grade_evidence(
        self,
        state: AgentState,
    ) -> AgentState:
        """使用确定性规则判断检索证据是否足够。"""

        try:
            retrieval_attempts = state["retrieval_attempts"]
            max_retrieval_attempts = state["max_retrieval_attempts"]
            min_relevance_score = state["min_relevance_score"]
        except KeyError as exc:
            return self._failure_update(
                failure_reason=("Agent state was incomplete during evidence grading."),
                error_type=type(exc).__name__,
            )

        results = state.get(
            "retrieval_results",
            [],
        )

        if results:
            highest_score = max(result.score for result in results)

            if highest_score >= min_relevance_score:
                return AgentState(
                    route=AgentRoute.ANSWER,
                    failure_reason=None,
                    error_type=None,
                )

        if retrieval_attempts < max_retrieval_attempts:
            return AgentState(
                route=AgentRoute.REWRITE_QUERY,
                failure_reason=("Retrieved evidence was insufficient."),
                error_type=None,
            )

        return AgentState(
            route=AgentRoute.REFUSE,
            failure_reason=(
                "Retrieved evidence remained insufficient after the maximum number of attempts."
            ),
            error_type=None,
        )

    async def rewrite_query(
        self,
        state: AgentState,
    ) -> AgentState:
        """使用聊天模型改写检索问题。"""

        try:
            original_question = state["original_question"]
            previous_query = state["retrieval_query"]

            response = await self._chat_provider.complete(
                [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=_REWRITE_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=(
                            "Original question:\n"
                            f"{original_question}\n\n"
                            "Previous search query:\n"
                            f"{previous_query}\n\n"
                            "Return an improved search query."
                        ),
                    ),
                ],
                json_mode=False,
            )

        except Exception as exc:
            return self._failure_update(
                failure_reason=("Query rewriting failed."),
                error_type=type(exc).__name__,
            )

        rewritten_query = response.content.strip().strip("\"'").strip()

        if not rewritten_query:
            return self._failure_update(
                failure_reason=("Chat provider returned an empty rewritten query."),
                error_type=("EmptyRewrittenQueryError"),
            )

        return AgentState(
            retrieval_query=rewritten_query,
            retrieval_results=[],
            route=AgentRoute.RETRIEVE,
            failure_reason=None,
            error_type=None,
        )

    async def answer(
        self,
        state: AgentState,
    ) -> AgentState:
        """使用现有回答服务生成结构化答案。"""

        try:
            answer = await self._answer_generator.answer(
                question=state["original_question"],
                results=state.get(
                    "retrieval_results",
                    [],
                ),
            )

        except Exception as exc:
            return self._failure_update(
                failure_reason=("Evidence-grounded answering failed."),
                error_type=type(exc).__name__,
            )

        failure_reason = (
            "Answering service refused because the available evidence was insufficient."
            if answer.refused
            else None
        )

        return AgentState(
            answer=answer,
            route=AgentRoute.END,
            failure_reason=failure_reason,
            error_type=None,
        )

    def refuse(
        self,
        state: AgentState,
    ) -> AgentState:
        """达到最大检索次数后安全拒答。"""

        failure_reason = state.get(
            "failure_reason",
            "Retrieved evidence was insufficient.",
        )

        return AgentState(
            answer=EvidenceAnswer(
                answer=("当前知识库中没有足够证据回答该问题。"),
                citations=[],
                confidence=0.0,
                refused=True,
            ),
            route=AgentRoute.END,
            failure_reason=failure_reason,
            error_type=None,
        )

    def handle_error(
        self,
        state: AgentState,
    ) -> AgentState:
        """把内部错误转换成安全拒答。"""

        return AgentState(
            answer=EvidenceAnswer(
                answer=("知识助手暂时无法完成该请求，请稍后重试。"),
                citations=[],
                confidence=0.0,
                refused=True,
            ),
            route=AgentRoute.END,
            failure_reason=state.get(
                "failure_reason",
                "Agent execution failed.",
            ),
            error_type=state.get(
                "error_type",
                "UnknownAgentError",
            ),
        )

    @staticmethod
    def _failure_update(
        *,
        failure_reason: str,
        error_type: str,
        retrieval_attempts: int | None = None,
    ) -> AgentState:
        """构造不包含原始异常内容的失败状态。"""

        update = AgentState(
            route=AgentRoute.HANDLE_ERROR,
            failure_reason=failure_reason,
            error_type=error_type,
        )

        if retrieval_attempts is not None:
            update["retrieval_attempts"] = retrieval_attempts

        return update
