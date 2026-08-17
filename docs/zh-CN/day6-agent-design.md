# 第六天：LangGraph 知识 Agent 架构设计

## 1. 目标

把当前固定执行的 RAG 流程：

```text
retrieve → answer
```

升级为带有共享状态、条件判断和有限重试能力的 LangGraph 工作流：

```text
START
  ↓
retrieve
  ↓
grade_evidence
  ├─ 证据充分 → answer → END
  ├─ 证据不足且可以重试 → rewrite_query → retrieve
  └─ 重试次数耗尽 → refuse → END
```

## 2. 设计原则

1. 保留现有 `RetrievalService` 和 `AnsweringService`。
2. LangGraph 只负责编排，不重新实现检索和回答。
3. 每个节点只读取当前状态并返回状态更新。
4. 设置最大检索次数，禁止无限循环。
5. 检索失败和回答失败进入统一错误处理流程。
6. 证据不足时必须拒答，不能要求模型自行编造。
7. 后续使用 `thread_id` 隔离不同会话。
8. LangSmith 关闭时，本地 Agent 和测试必须正常运行。

## 3. 节点职责

### retrieve

读取 `retrieval_query`、`top_k` 和 `filters`，调用现有检索服务，保存检索结果并增加检索次数。

### grade_evidence

根据检索结果最高相似度、最低相关度阈值和剩余重试次数，选择回答、改写或拒答。

### rewrite_query

保留用户原始意图，将当前问题改写为更适合语义检索的查询，然后重新进入检索节点。

### answer

使用原始问题和最终检索证据调用现有 `AnsweringService`，生成带引用的结构化回答。

### refuse

当证据不足且检索次数耗尽时，返回不带引用的结构化拒答。

### handle_error

把内部异常转换成安全响应，同时在状态中保留错误类型和失败阶段。

## 4. 路由规则

```text
retrieve 成功             → grade_evidence
retrieve 异常             → handle_error
证据达到相关度阈值         → answer
证据不足且仍有剩余次数     → rewrite_query
证据不足且次数已经耗尽     → refuse
rewrite_query 成功         → retrieve
rewrite_query 异常         → handle_error
answer 成功               → END
answer 异常               → handle_error
```

## 5. 状态字段

| 字段 | 含义 |
|---|---|
| `question_history` | 同一会话中的问题历史 |
| `original_question` | 用户最初的问题，后续节点不得覆盖 |
| `retrieval_query` | 当前用于向量检索的问题 |
| `top_k` | 最多返回多少个 Chunk |
| `filters` | 文档类型、文件名等检索过滤条件 |
| `retrieval_attempts` | 已经执行的检索次数 |
| `max_retrieval_attempts` | 最大检索次数 |
| `min_relevance_score` | 证据最低相关度阈值 |
| `retrieval_results` | 当前检索到的证据 |
| `answer` | 最终结构化答案 |
| `route` | 下一步路由 |
| `failure_reason` | 当前失败原因 |
| `error_type` | 当前异常类型 |

## 6. 循环终止条件

Agent 必须在以下任意条件下终止：

1. 成功生成带合法引用的回答。
2. 证据不足且达到最大检索次数。
3. 检索、改写或回答过程中发生异常。

不允许无上限执行 `rewrite_query → retrieve`。

## 7. 任务一边界

任务一只定义架构、请求模型和共享状态，暂不实现：

- 真实检索节点；
- 查询改写节点；
- 回答节点；
- LangGraph 编译；
- Checkpointer；
- LangSmith；
- 真实 API 调用。

这些能力将在第六天后续任务中逐步实现。

## 8. 对外结果边界

LangGraph内部使用`AgentState`保存完整执行状态，但应用层不能直接把状态返回给调用方。

公开结果使用：

- `AgentOutcome`：区分回答、正常拒答和执行失败；
- `AgentEvidence`：只返回引用元数据，不返回Chunk正文；
- `KnowledgeAgentResult`：返回最终答案、检索次数、最终查询和证据摘要；
- `create_agent_result`：负责把内部状态转换为公开结果。

结果状态：

```text
answered：正常生成有引用回答
refused：程序正常运行，但证据不足
failed：检索、改写或回答过程发生异常
