# 第六天：LangGraph 知识 Agent 架构设计

## 1. 今日目标

前五天完成的是一条固定的确定性 RAG 流程：

```text
用户问题
→ 向量检索
→ 生成带引用的回答
```

第六天将其升级为具有自主路由、有限重试和安全失败能力的 LangGraph 知识 Agent：

```text
START
  ↓
retrieve
  ↓
grade_evidence
  ├─ 证据充分 → answer → END
  ├─ 证据不足且可以重试 → rewrite_query → retrieve
  └─ 重试次数耗尽 → refuse → END

任意节点异常
  ↓
handle_error
  ↓
END
```

本阶段不会推翻前五天已经实现的 RAG 服务，而是在现有服务外增加 Agent 编排层。

---

## 2. 架构分层

当前应用分为以下几层：

```text
调用方
  ↓
Knowledge Agent
  ├─ AgentState
  ├─ LangGraph节点
  ├─ 条件路由
  └─ 公开结果转换
  ↓
现有RAG应用服务
  ├─ RetrievalService
  └─ AnsweringService
  ↓
Provider与存储
  ├─ EmbeddingProvider
  ├─ ChatProvider
  └─ VectorStoreProvider
```

各层职责如下：

| 层级 | 职责 |
|---|---|
| Agent 层 | 状态管理、节点编排、条件判断、有限重试 |
| Retrieval 层 | 查询向量化、向量数据库检索、Top-k 排序 |
| Answering 层 | Prompt 构建、模型调用、JSON 校验、引用校验 |
| Provider 层 | 屏蔽 DeepSeek、硅基流动等外部 API 差异 |
| Vector Store 层 | 保存并检索文档 Chunk 向量 |

LangGraph 只负责编排，不重新实现检索和回答业务。

---

## 3. 设计原则

1. 保留现有 `RetrievalService` 和 `AnsweringService`。
2. LangGraph 只负责执行顺序和条件路由。
3. 每个节点只读取当前状态并返回需要修改的字段。
4. 设置最大检索次数，禁止无限循环。
5. 检索、改写和回答异常进入统一错误处理。
6. 证据不足时必须拒答，不能要求模型自行编造。
7. 原始问题和当前检索问题必须分开保存。
8. 内部状态不能直接作为公共 API 返回值。
9. 节点通过构造函数注入依赖，便于测试和替换。
10. LangSmith 关闭时，本地 Agent 和测试仍须正常运行。
11. 后续使用 `thread_id` 隔离不同用户和会话。
12. 日志及公开结果不得包含 API Key 或原始异常敏感信息。

---

## 4. Agent 执行流程

完整规划流程如下：

```text
START
  ↓
retrieve
  ↓
grade_evidence
  ├─ route=answer
  │    ↓
  │  answer
  │    ├─ 成功 → END
  │    └─ 异常 → handle_error → END
  │
  ├─ route=rewrite_query
  │    ↓
  │  rewrite_query
  │    ├─ 成功 → retrieve
  │    └─ 异常 → handle_error → END
  │
  └─ route=refuse
       ↓
     refuse
       ↓
      END
```

检索节点发生异常时：

```text
retrieve
  ↓
handle_error
  ↓
END
```

---

## 5. 节点职责

### 5.1 retrieve

`retrieve` 节点负责：

1. 读取当前 `retrieval_query`；
2. 读取 `top_k` 和 `filters`；
3. 构造 `RetrievalQuery`；
4. 调用 `Retriever.retrieve()`；
5. 增加 `retrieval_attempts`；
6. 保存 `retrieval_results`；
7. 将下一步路由设置为 `grade_evidence`。

正常结果：

```text
route=grade_evidence
```

异常结果：

```text
route=handle_error
error_type=<异常类型>
failure_reason=Knowledge retrieval failed.
```

节点不会把原始异常信息直接返回给用户。

### 5.2 grade_evidence

`grade_evidence` 使用确定性规则判断证据是否足够。

判断依据：

- 是否存在检索结果；
- 最高相似度是否达到 `min_relevance_score`；
- 当前检索次数是否小于 `max_retrieval_attempts`。

路由规则：

```text
最高分 >= 最低相关度阈值
→ answer

最高分 < 最低相关度阈值
并且仍有剩余检索次数
→ rewrite_query

证据不足
并且检索次数已经耗尽
→ refuse
```

该节点不调用大模型，因此判断过程可复现、可测试。

### 5.3 rewrite_query

`rewrite_query` 负责在证据不足时改写检索问题。

执行原则：

1. 保留用户原始意图；
2. 只改写检索查询；
3. 不回答用户问题；
4. 不添加原问题中不存在的事实；
5. 不返回 Markdown 或引号；
6. 使用 `ChatProvider` 调用聊天模型。

成功时：

```text
更新 retrieval_query
清空 retrieval_results
route=retrieve
```

失败或返回空字符串时：

```text
route=handle_error
```

`original_question` 在改写过程中不得被覆盖。

### 5.4 answer

`answer` 使用原始问题和最终检索证据生成结构化回答。

输入：

- `original_question`；
- `retrieval_results`。

依赖：

- `AnswerGenerator`；
- 当前实现中由 `AnsweringService`满足该接口。

输出：

- `EvidenceAnswer`；
- 合法引用；
- 置信度；
- 是否拒答。

成功时：

```text
保存 answer
route=end
```

异常时：

```text
route=handle_error
```

即使检索查询被改写，最终回答仍然必须针对用户的原始问题。

### 5.5 refuse

`refuse` 在达到最大检索次数后生成确定性的安全拒答：

```text
当前知识库中没有足够证据回答该问题。
```

拒答结果必须满足：

- `refused=True`；
- `citations=[]`；
- `confidence=0.0`；
- `route=end`。

该节点不再调用大模型，避免在证据不足时继续消耗 Token 或产生幻觉。

### 5.6 handle_error

`handle_error` 将内部异常转换为安全的结构化响应：

```text
知识助手暂时无法完成该请求，请稍后重试。
```

错误结果必须满足：

- `refused=True`；
- `citations=[]`；
- `confidence=0.0`；
- 保留安全的 `failure_reason`；
- 保留异常类型 `error_type`；
- 不包含原始异常详情；
- `route=end`。

---

## 6. 路由规则

| 当前节点 | 条件 | 下一节点 |
|---|---|---|
| `retrieve` | 检索成功 | `grade_evidence` |
| `retrieve` | 检索异常 | `handle_error` |
| `grade_evidence` | 证据充分 | `answer` |
| `grade_evidence` | 证据不足且可以重试 | `rewrite_query` |
| `grade_evidence` | 证据不足且次数耗尽 | `refuse` |
| `rewrite_query` | 改写成功 | `retrieve` |
| `rewrite_query` | 改写异常或结果为空 | `handle_error` |
| `answer` | 回答成功 | `END` |
| `answer` | 回答异常 | `handle_error` |
| `refuse` | 完成安全拒答 | `END` |
| `handle_error` | 完成安全错误响应 | `END` |

LangGraph 的图编译和条件边将在任务五实现。

---

## 7. Agent 状态设计

LangGraph 节点通过 `AgentState` 共享状态。

### 7.1 状态字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `question_history` | `list[str]` | 同一会话的问题历史 |
| `original_question` | `str` | 用户最初提交的问题 |
| `retrieval_query` | `str` | 当前用于向量检索的问题 |
| `top_k` | `int` | 最多返回多少个 Chunk |
| `filters` | `dict[str, str]` | 文件名、文档类型等过滤条件 |
| `retrieval_attempts` | `int` | 已执行的检索次数 |
| `max_retrieval_attempts` | `int` | 最大允许检索次数 |
| `min_relevance_score` | `float` | 最低证据相关度阈值 |
| `retrieval_results` | `list[SearchResult]` | 当前检索结果 |
| `answer` | `EvidenceAnswer \| None` | 最终结构化回答 |
| `route` | `AgentRoute` | 下一步路由 |
| `failure_reason` | `str \| None` | 安全的失败原因 |
| `error_type` | `str \| None` | 异常类型 |

### 7.2 原始问题与检索问题

必须区分：

```text
original_question
```

和：

```text
retrieval_query
```

例如用户原问题是：

```text
前五天实现的检索功能有什么作用？
```

改写后的检索问题可能是：

```text
个人知识助手 向量检索 RetrievalService 工作原理
```

Agent 使用改写后的问题检索，但最终回答必须针对原始问题。

### 7.3 初始状态

`create_initial_agent_state()` 负责把经过 Pydantic 校验的 `KnowledgeAgentRequest` 转换为初始状态。

初始值包括：

```text
original_question = 用户问题
retrieval_query = 用户问题
retrieval_attempts = 0
retrieval_results = []
answer = None
route = retrieve
failure_reason = None
error_type = None
```

---

## 8. Agent 请求配置

`KnowledgeAgentRequest` 是一次 Agent 执行的输入模型。

包含：

| 参数 | 默认值 | 约束 |
|---|---:|---|
| `question` | 无 | 不能为空 |
| `top_k` | `5` | `1～20` |
| `filters` | `{}` | 键和值不能为空 |
| `max_retrieval_attempts` | `2` | `1～5` |
| `min_relevance_score` | `0.35` | `-1.0～1.0` |

应用配置中对应字段为：

```text
PKA_AGENT_MAX_RETRIEVAL_ATTEMPTS=2
PKA_AGENT_MIN_RELEVANCE_SCORE=0.35
```

含义：

- 最大检索次数包含第一次原始问题检索；
- 默认值 `2` 表示最多执行一次原始问题检索和一次改写后检索；
- 最低相关度阈值用于判断证据是否足够；
- 阈值需要根据后续评测结果调整，不能只凭直觉决定。

---

## 9. Agent 路由枚举

`AgentRoute` 定义以下路由值：

| 路由 | 含义 |
|---|---|
| `retrieve` | 执行知识库检索 |
| `grade_evidence` | 判断证据是否充分 |
| `rewrite_query` | 改写检索查询 |
| `answer` | 生成带引用回答 |
| `refuse` | 生成安全拒答 |
| `handle_error` | 处理内部异常 |
| `end` | Agent 执行结束 |

节点不能使用任意字符串表达路由，统一使用 `AgentRoute` 可以降低拼写错误和非法状态风险。

---

## 10. 对外结果设计

LangGraph 内部使用 `AgentState` 保存完整执行状态，但应用层不能直接把内部状态返回给调用方。

公开结果使用：

- `AgentOutcome`；
- `AgentEvidence`；
- `KnowledgeAgentResult`；
- `create_agent_result()`。

### 10.1 AgentOutcome

结果分为三种：

```text
answered
refused
failed
```

含义：

| 结果 | 含义 |
|---|---|
| `answered` | 成功生成基于证据的回答 |
| `refused` | 程序正常执行，但证据不足 |
| `failed` | 检索、改写或回答过程发生异常 |

`refused` 和 `failed` 必须分开。

证据不足是正常业务结果，而内部异常是执行失败。

### 10.2 AgentEvidence

`AgentEvidence` 只返回引用证据的安全摘要：

- `chunk_id`；
- `file_name`；
- `page_number`；
- `section_path`；
- `score`。

不会直接返回完整 Chunk 正文，以降低知识库内容泄露风险。

### 10.3 KnowledgeAgentResult

公开结果包含：

- `outcome`；
- `answer`；
- `retrieval_attempts`；
- `final_retrieval_query`；
- `evidence`；
- `failure_reason`；
- `error_type`。

### 10.4 结果一致性

公开结果必须满足：

```text
answered
→ answer.refused=False
→ 至少包含一个合法引用
→ error_type=None

refused
→ answer.refused=True
→ citations=[]
→ error_type=None

failed
→ answer.refused=True
→ citations=[]
→ error_type不为空
→ failure_reason不为空
```

回答中的每个 `chunk_id` 都必须存在于公开的 `evidence` 中。

---

## 11. LangGraph 节点实现

任务四已经将现有 RAG 服务包装成可独立测试的 Agent 节点。

实现位置：

```text
src/personal_knowledge_assistant/agent/nodes.py
```

核心类：

```text
KnowledgeAgentNodes
```

### 11.1 节点依赖

`KnowledgeAgentNodes` 通过构造函数接收：

- `Retriever`：执行知识库检索；
- `AnswerGenerator`：依据证据生成结构化回答；
- `ChatProvider`：改写检索问题。

节点不自行创建 Provider 或 Service，这样可以：

- 使用 Fake 对象进行单元测试；
- 替换模型 Provider；
- 避免重复创建网络客户端；
- 由应用工厂统一装配依赖；
- 保持节点业务逻辑清晰。

### 11.2 节点状态更新

节点只返回自己修改的字段。

例如 `retrieve` 成功后返回：

```python
AgentState(
    retrieval_attempts=retrieval_attempts,
    retrieval_results=results,
    route=AgentRoute.GRADE_EVIDENCE,
    failure_reason=None,
    error_type=None,
)
```

它不需要重新返回 `original_question`、`top_k` 等未修改字段。

LangGraph 在执行时负责把节点返回值合并到共享状态中。

### 11.3 安全异常转换

业务节点捕获异常后，不把原始异常消息放入状态，而是只保存：

```text
failure_reason=安全的失败描述
error_type=异常类名称
route=handle_error
```

例如：

```text
failure_reason=Knowledge retrieval failed.
error_type=RuntimeError
```

以下内容不应该进入公开响应：

- API Key；
- 请求头；
- 数据库密码；
- 原始服务响应；
- 文件系统敏感路径；
- 完整异常堆栈。

### 11.4 节点与图的边界

任务四只实现节点，不直接决定 LangGraph 的条件边。

节点负责：

```text
执行业务逻辑
→ 更新状态
→ 写入 route
```

任务五的 Graph 负责：

```text
读取 route
→ 选择下一个节点
→ 控制循环与终止
```

这种拆分使节点可以脱离 LangGraph 单独测试。

---

## 12. 循环与终止条件

Agent 必须在以下任一条件下结束：

1. 成功生成带合法引用的回答；
2. 证据不足且达到最大检索次数；
3. 检索、改写或回答发生异常。

查询改写循环为：

```text
retrieve
→ grade_evidence
→ rewrite_query
→ retrieve
```

循环上限由以下配置控制：

```text
max_retrieval_attempts
```

例如：

```text
max_retrieval_attempts=2
```

最多执行：

```text
第1次：使用原始问题检索
第2次：使用改写后的问题检索
```

第二次仍然证据不足时进入 `refuse`，不能继续改写。

---

## 13. 测试策略

任务一至任务四采用分层单元测试。

### 13.1 状态模型测试

验证：

- 请求默认值；
- 问题不能为空；
- Top-k 边界；
- 最大检索次数边界；
- 相关度阈值边界；
- 初始状态字段；
- 状态历史 reducer。

### 13.2 公开结果测试

验证：

- 正常回答结果；
- 正常拒答结果；
- 异常失败结果；
- 引用必须存在于证据列表；
- 证据 ID 不得重复；
- 回答状态与 `outcome` 一致；
- 内部状态可安全转换为公开结果。

### 13.3 节点测试

任务四已经覆盖：

1. 检索请求参数正确传递；
2. 检索次数正确增加；
3. 检索异常进入错误处理；
4. 证据充分进入回答节点；
5. 证据不足进入查询改写；
6. 检索次数耗尽进入拒答；
7. 查询改写正确调用聊天模型；
8. 空改写结果进入错误处理；
9. 回答使用原始问题；
10. 回答异常进入错误处理；
11. 拒答和错误响应符合安全要求。

### 13.4 当前测试结果

任务五完成后的质量检查结果：

```text
Ruff format: passed
Ruff check: passed
mypy: passed
Agent node tests: 11 passed
Agent graph tests: 4 passed
Full test suite: 354 passed
```

测试数量会随着后续任务继续增加。

---

## 14. LangGraph 图编排实现

任务五已经在 `agent/graph.py` 中把六个独立节点连接为可执行状态图。

### 14.1 图入口

每次执行从 `retrieve` 节点开始：

```text
START → retrieve
```

### 14.2 条件路由

图通过读取节点写入的 `route` 字段选择下一节点：

```text
retrieve
  ├─ grade_evidence
  └─ handle_error

grade_evidence
  ├─ answer
  ├─ rewrite_query
  ├─ refuse
  └─ handle_error

rewrite_query
  ├─ retrieve
  └─ handle_error

answer
  ├─ END
  └─ handle_error
```

### 14.3 终止节点

以下节点完成后进入 LangGraph 的 `END`：

- `answer` 成功；
- `refuse` 完成安全拒答；
- `handle_error` 完成安全异常响应。

### 14.4 查询改写循环

证据不足但仍有剩余次数时执行：

```text
grade_evidence
→ rewrite_query
→ retrieve
→ grade_evidence
```

循环次数由 `max_retrieval_attempts` 控制，不允许无限执行。

### 14.5 图编排测试

任务五验证了四条端到端图路径：

1. 第一次检索证据充分并成功回答；
2. 第一次证据不足，改写问题后成功回答；
3. 达到最大检索次数后安全拒答；
4. 检索异常进入统一安全错误处理。

---

## 15. 当前实现进度

| 任务 | 内容 | 状态 |
|---|---|---|
| 任务一 | Agent 请求、状态与路由设计 | 已完成 |
| 任务二 | LangGraph、LangSmith 依赖与 Agent 配置 | 已完成 |
| 任务三 | Agent 公开结果模型 | 已完成 |
| 任务四 | 检索、评分、改写、回答、拒答及错误节点 | 已完成 |
| 任务五 | 构建并编译 LangGraph | 已完成 |
| 任务六 | Agent Service、Checkpointer 与会话隔离 | 待完成 |
| 任务七 | LangSmith Trace、端到端验证与收尾 | 待完成 |

---

## 16. 任务六设计边界

任务六负责：

1. 允许 Graph 在编译时接收 Checkpointer；
2. 默认使用 LangGraph `InMemorySaver`；
3. 实现 `KnowledgeAgentService`；
4. 使用 `thread_id` 隔离不同会话；
5. 将 Graph 内部状态转换为 `KnowledgeAgentResult`；
6. 提供问题历史读取能力；
7. 将 Agent Service 接入应用工厂；
8. 测试同会话状态累积与不同会话隔离。

任务六不负责：

- 跨进程或服务重启后的持久记忆；
- PostgreSQL Checkpointer；
- 用户身份认证；
- 会话授权；
- LangSmith 在线 Trace；
- Web API 或命令行界面。

`InMemorySaver` 只适用于本地开发、学习和单进程测试。生产环境必须替换为持久化 Checkpointer。

---

## 17. 任务六目标调用方式

任务六完成后，调用方不再直接操作 Graph：

```python
result = await agent_service.run(
    request,
    thread_id="user-13547-session-1",
)
```

Service 内部负责：

```text
校验 thread_id
→ 创建初始 AgentState
→ 使用 thread_id 调用 graph.ainvoke
→ 从 Checkpointer 保存或恢复状态
→ 转换为 KnowledgeAgentResult
→ 返回安全公开结果
```

同一个 `thread_id` 的问题历史会继续累积；不同 `thread_id` 的状态必须完全隔离。

---

## 18. 任务七规划

任务七将在 Agent 可以稳定执行和隔离会话后，补充：

- LangSmith Trace；
- 节点执行可观测性；
- Token、耗时和错误记录；
- 真实 DeepSeek 与硅基流动端到端验证；
- 第六天完整质量门禁；
- GitHub 上传与阶段总结。
