from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from personal_knowledge_assistant.agent.nodes import (
    KnowledgeAgentNodes,
)
from personal_knowledge_assistant.agent.state import (
    AgentRoute,
    AgentState,
)

KnowledgeAgentGraph = CompiledStateGraph[
    AgentState,
    None,
    AgentState,
    AgentState,
]


def _select_route(
    state: AgentState,
) -> AgentRoute:
    """读取节点写入状态的下一步路由。

    如果节点没有写入 route，则统一进入安全错误处理节点。
    """

    return state.get(
        "route",
        AgentRoute.HANDLE_ERROR,
    )


def create_knowledge_agent_graph(
    *,
    nodes: KnowledgeAgentNodes,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> KnowledgeAgentGraph:
    """创建并编译个人知识助手 LangGraph。

    Args:
        nodes: 已完成依赖注入的 Agent 节点集合。
        checkpointer: 可选的 LangGraph 状态保存器。

    Returns:
        已编译的知识 Agent 状态图。
    """

    builder = StateGraph[
        AgentState,
        None,
        AgentState,
        AgentState,
    ](
        AgentState,
        input_schema=AgentState,
        output_schema=AgentState,
    )

    builder.add_node(
        AgentRoute.RETRIEVE.value,
        nodes.retrieve,
    )
    builder.add_node(
        AgentRoute.GRADE_EVIDENCE.value,
        nodes.grade_evidence,
    )
    builder.add_node(
        AgentRoute.REWRITE_QUERY.value,
        nodes.rewrite_query,
    )
    builder.add_node(
        AgentRoute.ANSWER.value,
        nodes.answer,
    )
    builder.add_node(
        AgentRoute.REFUSE.value,
        nodes.refuse,
    )
    builder.add_node(
        AgentRoute.HANDLE_ERROR.value,
        nodes.handle_error,
    )

    builder.add_edge(
        START,
        AgentRoute.RETRIEVE.value,
    )

    builder.add_conditional_edges(
        AgentRoute.RETRIEVE.value,
        _select_route,
        {
            AgentRoute.GRADE_EVIDENCE: (AgentRoute.GRADE_EVIDENCE.value),
            AgentRoute.HANDLE_ERROR: (AgentRoute.HANDLE_ERROR.value),
        },
    )

    builder.add_conditional_edges(
        AgentRoute.GRADE_EVIDENCE.value,
        _select_route,
        {
            AgentRoute.ANSWER: (AgentRoute.ANSWER.value),
            AgentRoute.REWRITE_QUERY: (AgentRoute.REWRITE_QUERY.value),
            AgentRoute.REFUSE: (AgentRoute.REFUSE.value),
            AgentRoute.HANDLE_ERROR: (AgentRoute.HANDLE_ERROR.value),
        },
    )

    builder.add_conditional_edges(
        AgentRoute.REWRITE_QUERY.value,
        _select_route,
        {
            AgentRoute.RETRIEVE: (AgentRoute.RETRIEVE.value),
            AgentRoute.HANDLE_ERROR: (AgentRoute.HANDLE_ERROR.value),
        },
    )

    builder.add_conditional_edges(
        AgentRoute.ANSWER.value,
        _select_route,
        {
            AgentRoute.END: END,
            AgentRoute.HANDLE_ERROR: (AgentRoute.HANDLE_ERROR.value),
        },
    )

    builder.add_edge(
        AgentRoute.REFUSE.value,
        END,
    )
    builder.add_edge(
        AgentRoute.HANDLE_ERROR.value,
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name="personal-knowledge-agent",
    )
