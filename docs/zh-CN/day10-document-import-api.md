# 第十天：文档上传与自动索引 API

## 完成目标

Day 10 将已有的文档解析、切块、Embedding 与 pgvector 存储能力接入 FastAPI。调用一次上传接口后，服务会按顺序完成：

```text
multipart 文件上传
  → 校验文件名、扩展名和大小
  → 原子保存到 data/uploads
  → 解析 TXT / Markdown / PDF
  → 切块并生成 Embedding
  → 写入 PostgreSQL + pgvector
  → 返回导入和索引状态
```

接口为 `POST /api/v1/documents/import`，支持 `.txt`、`.md`、`.markdown` 和 `.pdf`。默认限制为 10 MiB。

## 1. 准备配置

确认项目根目录 `.env` 至少包含以下配置。API Key 只保存在本机 `.env`，不要提交到 Git：

```dotenv
PKA_EMBEDDING_PROVIDER=siliconflow
PKA_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
PKA_EMBEDDING_API_KEY=替换为你的Key
PKA_EMBEDDING_DIMENSION=1024

PKA_VECTOR_STORE_PROVIDER=pgvector
PKA_DATABASE_URL=postgresql://pka:pka-local-password@localhost:5432/personal_knowledge
PKA_DATABASE_SCHEMA=pka

PKA_UPLOAD_DIRECTORY=data/uploads
PKA_UPLOAD_MAX_BYTES=10485760
```

如果修改过 Embedding 维度，数据库表的 `vector(1024)` 也必须使用相同维度。

## 2. 安装依赖并启动数据库

在项目根目录执行：

```powershell
uv sync
docker compose up -d postgres
docker compose ps
uv run python scripts/check_database.py
```

`docker compose ps` 中 `pka-postgres` 应显示为 `Up` 和 `healthy`；数据库检查脚本应确认 PostgreSQL、pgvector 扩展及表结构可用。

## 3. 运行自动化测试

普通测试不调用真实数据库和外部 Embedding API：

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

额外运行真实 PostgreSQL/pgvector 集成测试：

```powershell
$env:PKA_RUN_POSTGRES_TESTS="1"
uv run pytest tests/test_pgvector_store.py
Remove-Item Env:PKA_RUN_POSTGRES_TESTS
```

集成测试使用确定性本地向量，不消耗 Embedding API 额度，并会清理自己的测试数据。

## 4. 启动 API

保持 PostgreSQL 容器运行，然后执行：

```powershell
uv run uvicorn personal_knowledge_assistant.api:create_api_app --factory --host 127.0.0.1 --port 8000
```

出现 `Application startup complete` 后不要关闭该窗口。浏览器可打开：

- Swagger：<http://127.0.0.1:8000/docs>
- 存活检查：<http://127.0.0.1:8000/health/live>
- 就绪检查：<http://127.0.0.1:8000/health/ready>

## 5. 手动上传文档

另开一个 PowerShell，并先进入项目根目录，否则 `curl.exe` 找不到相对路径中的文件：

```powershell
Set-Location D:\Python\Agent\personal-knowledge-assistant

curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/import" `
  -H "accept: application/json" `
  -F "file=@evaluation/documents/rag.md;type=text/markdown"
```

成功时返回 HTTP 200，响应示例：

```json
{
  "document_id": "file:///.../data/uploads/rag.md",
  "file_name": "rag.md",
  "content_hash": "...",
  "import_status": "created",
  "indexing_status": "indexed",
  "chunk_count": 3,
  "deactivated_chunk_count": 0
}
```

相同内容再次上传时，`import_status` 为 `unchanged`、`indexing_status` 为 `skipped`。修改文件后再上传时，状态为 `updated`，旧 Chunk 会停用，新 Chunk 会写入。

## 6. 验收标准

- 上传 TXT、Markdown 或 PDF 返回 200，文件出现在 `data/uploads`。
- 超过配置大小返回 413。
- 不支持的扩展名返回 415。
- 空文档或无法解析的文档返回 422。
- 路径穿越或 Windows 保留文件名返回 400。
- 内部错误返回安全的 503，不向客户端泄露数据库或供应商异常。
- PostgreSQL 集成测试能够检索到刚上传文档的 Chunk。

以上全部通过后，Day 10 即完成。下一阶段可实现持久化文档目录与服务重启后的增量导入状态。
