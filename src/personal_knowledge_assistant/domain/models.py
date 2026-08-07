from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


# 模型角色定义
class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# 模型输入消息
class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


# 模型输出及 Token 用量
class ChatResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)


# 以后进入向量库的文档片段
class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 检索结果
class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
