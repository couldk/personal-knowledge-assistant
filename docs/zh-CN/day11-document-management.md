# 第十一天：持久化文档目录与管理 API

## 完成目标

Day 11 使用 PostgreSQL 中的 `documents`、`document_versions` 和 `document_chunks` 作为文档状态的持久化来源，解决 Day 10 仅依赖进程内 Catalog 的限制，并提供文档列表、详情、软删除和重新索引接口。

```text
上传文档
  → 解析得到稳定 document_id 与 content_hash
  → 查询 PostgreSQL 活动版本
  → created / unchanged / updated
  → 必要时写入新向量

文档管理 API
  → 列表 / 详情 / 软删除 / 强制重新索引
```

## API

| 方法 | 路径 | 功能 |
|---|---|---|
| `GET` | `/api/v1/documents` | 分页查询未删除文档 |
| `GET` | `/api/v1/documents/{document_key}` | 查询文档详情 |
| `DELETE` | `/api/v1/documents/{document_key}` | 软删除文档及活动向量 |
| `POST` | `/api/v1/documents/{document_key}/reindex` | 从保存的源文件强制重新索引 |

`document_key` 是 PostgreSQL `documents.id` 的 UUID。所有查询和修改同时使用当前 `tenant_id` 过滤，不能跨租户读取或删除文档。

## 持久化导入判断

生产环境的 `DocumentImportService` 会通过 `PgVectorStore.get_document_by_id()` 查询活动内容哈希：

- 数据库不存在活动版本：`created`。
- 活动哈希与新文件相同：`unchanged`，跳过 Embedding。
- 活动哈希与新文件不同：`updated`，停用旧版本并写入新版本。

因此即使 FastAPI 重启、内存 Catalog 被清空，重复上传和更新判断仍然正确。

## 启动和手动验证

在项目根目录启动数据库：

```powershell
docker compose up -d postgres
docker compose ps
uv run python scripts/check_database.py
```

启动 API：

```powershell
uv run uvicorn personal_knowledge_assistant.api:create_api_app --factory --host 127.0.0.1 --port 8000
```

上传测试文档：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/import" `
  -H "accept: application/json" `
  -F "file=@evaluation/documents/rag.md;type=text/markdown"
```

查询文档列表：

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents?limit=20&offset=0"
```

从列表响应复制 `document_key`，然后查询详情：

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents/替换为document_key"
```

重新索引：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/替换为document_key/reindex"
```

软删除：

```powershell
curl.exe -X DELETE -i "http://127.0.0.1:8000/api/v1/documents/替换为document_key"
```

删除成功返回 `204 No Content`。删除后，该文档不会出现在默认列表中，其活动版本和 Chunk 会被停用，也不会参与向量检索。

## 自动化验收

运行全部普通测试与静态检查：

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

启用真实 PostgreSQL 测试：

```powershell
$env:PKA_RUN_POSTGRES_TESTS="1"
uv run pytest tests/test_pgvector_store.py
Remove-Item Env:PKA_RUN_POSTGRES_TESTS
```

真实数据库测试覆盖：

- 文档列表与详情。
- 软删除以及重复删除。
- 删除后向量不可检索。
- 新建另一套导入服务后，相同内容仍为 `unchanged`。
- 新建另一套导入服务后，变更内容仍为 `updated`。

## 完成标准

- 所有文档管理接口具有稳定响应和安全错误信息。
- 分页参数无效时返回 422。
- 文档不存在或已删除时返回 404。
- 内部数据库错误返回 503，不泄露异常内容。
- 服务重启不影响增量导入判断。
- 软删除同时停用活动版本和活动 Chunk。
- 强制重新索引会重新生成并写入向量。
- Ruff、mypy、普通测试和 PostgreSQL 集成测试全部通过。
