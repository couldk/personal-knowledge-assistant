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


class KnowledgeAgentService:
    """个人知识 Agent 应用服务。

    负责：

    1. 校验 thread_id；
    2. 创建初始 Agent 状态；
    3. 调用已经编译的 LangGraph；
    4. 将内部状态转换为公开结果；
    5. 读取指定会话的问题历史。
    """

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

        config = self._create_thread_config(
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

        config = self._create_thread_config(
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

    @staticmethod
    def _create_thread_config(
        thread_id: str,
    ) -> RunnableConfig:
        """创建 LangGraph 会话配置。"""

        normalized_thread_id = thread_id.strip()

        if not normalized_thread_id:
            raise ValueError("thread_id cannot be empty.")

        return RunnableConfig(
            configurable={
                "thread_id": normalized_thread_id,
            }
        )
