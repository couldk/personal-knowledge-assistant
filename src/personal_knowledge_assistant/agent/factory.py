from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
)
from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from personal_knowledge_assistant.agent.graph import (
    create_knowledge_agent_graph,
)
from personal_knowledge_assistant.agent.nodes import (
    KnowledgeAgentNodes,
)
from personal_knowledge_assistant.agent.service import (
    KnowledgeAgentService,
)
from personal_knowledge_assistant.providers.base import (
    ChatProvider,
)
from personal_knowledge_assistant.querying.base import (
    AnswerGenerator,
    Retriever,
)


def create_knowledge_agent_service(
    *,
    retriever: Retriever,
    answer_generator: AnswerGenerator,
    chat_provider: ChatProvider,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> KnowledgeAgentService:
    """创建完成依赖装配的知识 Agent Service。"""

    selected_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

    nodes = KnowledgeAgentNodes(
        retriever=retriever,
        answer_generator=answer_generator,
        chat_provider=chat_provider,
    )

    graph = create_knowledge_agent_graph(
        nodes=nodes,
        checkpointer=selected_checkpointer,
    )

    return KnowledgeAgentService(
        graph=graph,
    )
