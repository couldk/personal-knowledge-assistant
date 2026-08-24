from hashlib import sha256
from typing import cast

from langchain_core.runnables import RunnableConfig

from personal_knowledge_assistant.agent.graph import (
    KnowledgeAgentGraph,
)
from personal_knowledge_assistant.agent.models import (
    KnowledgeAgentResult,
    create_agent_result,
)
from personal_knowledge_assistant.agent.state import (
    AgentState,
    KnowledgeAgentRequest,
    create_initial_agent_state,
)


def create_agent_run_config(
    thread_id: str,
) -> RunnableConfig:
    """创建 LangGraph 执行和 LangSmith Trace 配置。

    thread_id 用于 Checkpointer 会话隔离。

    Trace 元数据只保存 thread_id 的哈希值，
    不把调用方提供的原始会话标识放进 metadata。
    """

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise ValueError("thread_id cannot be empty.")

    thread_id_hash = sha256(normalized_thread_id.encode("utf-8")).hexdigest()

    return RunnableConfig(
        configurable={
            "thread_id": normalized_thread_id,
        },
        run_name="knowledge-agent-run",
        tags=[
            "personal-knowledge-assistant",
            "langgraph",
            "day6",
        ],
        metadata={
            "component": "knowledge-agent",
            "thread_id_hash": thread_id_hash,
        },
    )


class KnowledgeAgentService:
    """个人知识 Agent 应用服务。"""

    def __init__(
        self,
        *,
        graph: KnowledgeAgentGraph,
    ) -> None:
        self._graph = graph

    async def run(
        self,
        request: KnowledgeAgentRequest,
        *,
        thread_id: str,
    ) -> KnowledgeAgentResult:
        """执行一次知识 Agent 请求。"""

        config = create_agent_run_config(
            thread_id,
        )

        initial_state = create_initial_agent_state(
            request,
        )

        final_state = cast(
            AgentState,
            await self._graph.ainvoke(
                initial_state,
                config=config,
            ),
        )

        return create_agent_result(
            final_state,
        )

    async def get_question_history(
        self,
        *,
        thread_id: str,
    ) -> list[str]:
        """读取指定会话已经保存的问题历史。"""

        config = create_agent_run_config(
            thread_id,
        )

        snapshot = await self._graph.aget_state(
            config,
        )

        if not snapshot.values:
            return []

        state = cast(
            AgentState,
            snapshot.values,
        )

        return list(
            state.get(
                "question_history",
                [],
            )
        )
