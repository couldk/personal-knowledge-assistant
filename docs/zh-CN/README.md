# 个人知识助手：中文开发指南

## 项目目标

构建一个以证据为中心的个人知识助手，支持 Markdown、PDF 和纯文本导入，能够基于私有知识库回答问题、返回准确引用、在证据不足时拒答，并安全地维护用户级长期记忆。

建议开发周期为两周，每周投入 8～12 小时。本项目的重点不是做一个聊天页面，而是跑通知识导入、检索、Agent 编排、记忆、评测和部署的完整闭环。

## 推荐技术栈

- LangGraph：意图路由、检索、回答、验证、修复和状态持久化。
- LlamaIndex：文档加载、结构化切分、索引与检索。
- PostgreSQL + pgvector：生产存储；第一个本地版本可使用 Chroma。
- FastAPI：提供导入、查询和记忆管理接口。
- LangSmith：记录检索、模型和图节点 Trace，并运行回归评测。

## MVP 范围

第一版只实现四件事：导入文件、知识问答、引用校验和证据不足拒答。完成这些能力后，再加入多轮状态、用户偏好记忆、混合检索和 reranker。

建议的核心流程：

```text
用户问题
  → 判断是否需要知识检索
  → 检索候选片段
  → 检查证据覆盖度
  → 基于证据生成结构化答案
  → 校验引用是否来自本次检索
  → 返回答案或拒答
```

## 两周执行计划

### 第一周：建立可评测的 RAG 基线

1. 初始化 Python 3.12、uv、Ruff、mypy 和 pytest。
2. 定义模型、Embedding 和向量库适配接口，避免业务代码直接依赖供应商 SDK。
3. 实现文档哈希、解析、结构化切分、元数据保留和增量更新。
4. 建立至少 30 条问题的初始黄金集，再开始调整 chunk size 和 top-k。
5. 实现结构化回答：`answer`、`citations`、`confidence`、`refused`。
6. 使用确定性代码校验引用 ID，模型不得引用本次未检索到的片段。

### 第二周：加入 Agent、记忆和服务化

1. 使用 LangGraph 实现意图分类、检索、生成、校验和一次有界修复。
2. 按 `thread_id` 保存短期状态，按 `tenant_id/user_id` 隔离长期记忆。
3. 长期记忆只保存经过校验的事实或偏好，不自动永久保存完整对话。
4. 根据第一周失败样本决定是否加入关键词检索、元数据过滤或 reranker。
5. 提供导入、查询、文档状态、记忆查看/更正/删除接口。
6. 将黄金集扩充到 50 条并生成最终实验报告。

## 建议代码模块

```text
app/
├─ api/          # FastAPI 路由
├─ graph/        # LangGraph 状态和节点
├─ ingestion/    # 加载、切分、去重、更新
├─ retrieval/    # 检索和 rerank
├─ memory/       # 短期与长期记忆
├─ providers/    # 模型和 embedding 适配器
└─ schemas/      # Pydantic 输入输出
```

## 验收门槛

- 50 条黄金问题上 Recall@5 不低于 85%。
- 引用正确率不低于 90%。
- 无答案问题拒答率不低于 90%。
- 重新导入更新后的文件时，不存在仍处于活动状态的旧片段。
- 用户 A 的记忆在任何情况下都不能被用户 B 检索。
- 服务重启后能够从 Checkpoint 继续同一个会话。

## 开发时重点观察

当结果不好时，先区分是解析、切分、检索、上下文组织还是生成问题。不要只改 Prompt。向量数据库只是长期记忆和 RAG 的存储组件，不等于完整的记忆管理策略。

## 完整中文文档

- [需求说明](01-requirements.md)
- [架构设计](02-architecture.md)
- [实施计划](03-implementation-plan.md)
- [测试与验收](04-testing-and-acceptance.md)
- [部署与运维](05-deployment-and-operations.md)
- [第八天：PostgreSQL 持久化基础](day8-persistence-foundation.md)
- [第九天：PostgreSQL + pgvector 向量存储](day9-pgvector-store.md)

[English documentation](../en/README.md) · [返回项目主页](../../README.md)
