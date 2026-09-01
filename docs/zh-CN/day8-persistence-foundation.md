# 第八天：持久化基础设施

## 1. 目标

为个人知识助手建立可复现的PostgreSQL和pgvector开发环境，
并定义文档、文档版本、文档片段和用户长期记忆的数据库结构。

本任务只建立持久化基础，不实现PgVectorStore业务适配器。

## 2. 本地环境

本地数据库通过Docker Compose运行：

- PostgreSQL 17；
- pgvector 0.8.6；
- 数据库：personal_knowledge；
- 应用Schema：pka；
- 本地端口：5432。

## 3. 数据表

### documents

保存逻辑文档及当前状态。

### document_versions

保存文档的不同内容哈希版本，同一文档最多只有一个活动版本。

### document_chunks

保存可检索文档片段、来源元数据和1024维Embedding。

### user_memories

为后续长期记忆功能预留结构，目前尚未接入Agent。

## 4. 向量约束

当前Embedding模型：

```text
Qwen/Qwen3-Embedding-0.6B