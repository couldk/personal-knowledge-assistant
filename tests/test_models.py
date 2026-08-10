from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    MessageRole,
)


def test_chat_message() -> None:
    message = ChatMessage(
        role=MessageRole.USER,
        content="What is RAG?",
    )

    assert message.role == MessageRole.USER
    assert message.content == "What is RAG?"
