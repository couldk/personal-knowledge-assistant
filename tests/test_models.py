from personal_knowledge_assistant.domain.models import (
    ChatMessage,
    DocumentChunk,
    MessageRole,
    SearchResult,
)


def test_chat_message() -> None:
    message = ChatMessage(
        role=MessageRole.USER,
        content="What is RAG?",
    )

    assert message.role == MessageRole.USER
    assert message.content == "What is RAG?"


def test_search_result_contains_source_chunk() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-001",
        document_id="document-001",
        text="RAG combines retrieval with generation.",
        metadata={"source": "rag.md"},
    )

    result = SearchResult(chunk=chunk, score=0.95)

    assert result.chunk.document_id == "document-001"
    assert result.chunk.metadata["source"] == "rag.md"
