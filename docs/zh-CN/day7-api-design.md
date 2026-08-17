# 第七天：FastAPI服务化

## 1. 目标

把现有`KnowledgeAgentService`暴露为HTTP API，使浏览器、前端和其他服务可以调用知识Agent。

## 2. 架构

```text
HTTP客户端
→ FastAPI
→ API请求校验
→ KnowledgeAgentService
→ LangGraph
→ RetrievalService
→ AnsweringService
→ KnowledgeAgentResult
→ HTTP JSON响应
```

## 3. 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health/live` | 进程存活检查 |
| GET | `/health/ready` | Agent依赖就绪检查 |
| POST | `/api/v1/agent/query` | 执行Agent查询 |
| GET | `/api/v1/agent/threads/{thread_id}/history` | 查询会话问题历史 |

## 4. 请求模型

查询请求包括：

- `thread_id`；
- `question`；
- `top_k`；
- `filters`；
- `max_retrieval_attempts`；
- `min_relevance_score`。

API层负责格式和范围校验，领域层继续使用`KnowledgeAgentRequest`。

## 5. 返回模型

查询接口直接返回`KnowledgeAgentResult`。

结果包括：

- `outcome`；
- `answer`；
- `retrieval_attempts`；
- `final_retrieval_query`；
- `evidence`；
- `failure_reason`；
- `error_type`。

## 6. 生命周期

FastAPI使用`lifespan`在应用启动时创建一次共享应用服务。

不会为每个HTTP请求重复创建：

- Chat Provider；
- Embedding Provider；
- Vector Store；
- Agent Graph；
- Checkpointer。

## 7. 错误边界

请求格式错误返回`422`。

未处理的Agent服务异常返回：

```text
503 Service Unavailable
```

客户端不会收到：

- API Key；
- 数据库密码；
- Python堆栈；
- Provider原始错误；
- 服务内部路径。

Agent正常拒答仍返回`200`，并通过：

```text
outcome=refused
```

表达业务结果。

## 8. 会话状态

`thread_id`传递给LangGraph Checkpointer。

同一个`thread_id`共享问题历史，不同`thread_id`相互隔离。

当前使用`InMemorySaver`，服务重启后历史会丢失。

## 9. 测试范围

API测试覆盖：

1. 存活和就绪检查；
2. 合法Agent查询；
3. 非法请求返回422；
4. 会话历史查询；
5. 内部异常返回安全503；
6. 响应不泄露原始异常。

## 10. 当前限制

任务八暂不实现：

- 用户认证；
- 会话授权；
- 请求限流；
- CORS策略；
- PostgreSQL持久化Checkpointer；
- 文档上传接口；
- Docker部署；
- 多实例状态共享。

这些能力将在后续生产化任务中实现。