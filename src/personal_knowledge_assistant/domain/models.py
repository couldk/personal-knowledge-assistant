from enum import StrEnum

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """聊天消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """发送给聊天模型的一条消息。"""

    role: MessageRole
    content: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """聊天模型返回的内容和Token用量。"""

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
