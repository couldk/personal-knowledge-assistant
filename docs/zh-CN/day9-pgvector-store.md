# 第九天：PostgreSQL + pgvector 向量存储

## 1. 完成状态

Day 9 已完成 PostgreSQL 向量存储实现，并把它接入应用工厂和 FastAPI 生命周期。

当前验证结果：

```text
372 passed
All checks passed!
Success: no issues found
```

其中 3 个测试会真实连接 Docker 中的 PostgreSQL，覆盖写入、向量检索、过滤和文档停用。

## 2. 本日目标

Day 8 只建立数据库、扩展、Schema、表和索引。Day 9 在此基础上完成应用层接入：

1. 使用异步连接池连接 PostgreSQL；
2. 使用 pgvector 保存 1024 维 Embedding；
3. 批量新增或更新文档 Chunk；
4. 使用余弦距离进行语义检索；
5. 支持元数据过滤；
6. 支持停用旧文档 Chunk；
7. 在应用启动和关闭时管理连接池；
8. 保留内存向量库，方便单元测试；
9. 增加真实 PostgreSQL 集成测试。

Day 9 不包含用户认证、文档上传接口、长期记忆管理和 PostgreSQL Checkpointer，这些属于后续任务。

## 3. 整体架构

```text
FastAPI
  → ApplicationServices
  → create_vector_store(settings)
      ├─ provider=memory   → InMemoryVectorStore
      └─ provider=pgvector → PgVectorStore
                               → psycopg异步连接池
                               → PostgreSQL 17
                               → pgvector扩展
                               → pka.document_chunks
```

写入流程：

```text
DocumentChunk + Embedding
  → 批次校验
  → 写入/更新 documents
  → 写入/更新 document_versions
  → 写入/更新 document_chunks
  → 提交事务
```

检索流程：

```text
问题Embedding
  → 校验向量
  → 添加tenant和active过滤
  → 添加可选元数据过滤
  → pgvector余弦距离排序
  → 转换为SearchResult
```

## 4. 相关文件

| 文件 | 作用 |
|---|---|
| `compose.yaml` | 定义 PostgreSQL + pgvector 容器 |
| `infra/postgres/init/001_initial_schema.sql` | 创建扩展、Schema、表和索引 |
| `scripts/check_database.py` | 检查数据库版本、扩展、表和向量维度 |
| `src/personal_knowledge_assistant/config.py` | 数据库和连接池配置 |
| `src/personal_knowledge_assistant/vector_store/postgres.py` | PostgreSQL 向量存储实现 |
| `src/personal_knowledge_assistant/vector_store/factory.py` | 根据配置选择向量存储 |
| `src/personal_knowledge_assistant/vector_store/__init__.py` | 导出 `PgVectorStore` |
| `src/personal_knowledge_assistant/api/app.py` | 管理连接池生命周期 |
| `tests/test_memory_vector_store.py` | 工厂和内存实现测试 |
| `tests/test_pgvector_store.py` | PostgreSQL 集成测试 |
| `tests/conftest.py` | Windows 异步事件循环兼容配置 |
| `pyproject.toml` | 依赖和 pytest marker |

## 5. 前置条件

开始前需要：

- Python 3.12；
- uv；
- Docker Desktop；
- WSL 2；
- Docker Engine 处于 Running 状态；
- Day 8 的 `compose.yaml` 和初始化 SQL 位于项目目录。

检查版本：

```powershell
python --version
uv --version
docker version
docker compose version
wsl --status
```

检查关键文件：

```powershell
Test-Path .\compose.yaml
Test-Path .\infra\postgres\init\001_initial_schema.sql
Test-Path .\scripts\check_database.py
```

三条命令都应返回 `True`。

## 6. 安装依赖

`pyproject.toml` 应包含：

```toml
dependencies = [
    # 其他依赖省略
    "pgvector>=0.5.0,<0.6",
    "psycopg[binary,pool]>=3.3.4,<4",
]
```

同步环境：

```powershell
$env:UV_LINK_MODE = "copy"
uv sync --all-groups
```

`UV_LINK_MODE=copy` 用于避免 Windows 上的硬链接警告，不会改变程序行为。

## 7. 环境变量

`.env` 中与数据库有关的配置应为：

```dotenv
# Docker中的PostgreSQL
PKA_POSTGRES_USER=pka
PKA_POSTGRES_PASSWORD=pka-local-password
PKA_POSTGRES_DB=personal_knowledge
PKA_POSTGRES_PORT=5432

# 应用中的向量存储
PKA_VECTOR_STORE_PROVIDER=pgvector
PKA_DATABASE_URL=postgresql://pka:pka-local-password@localhost:5432/personal_knowledge
PKA_DATABASE_SCHEMA=pka
PKA_DATABASE_CONNECT_TIMEOUT_SECONDS=10
PKA_DATABASE_POOL_MIN_SIZE=1
PKA_DATABASE_POOL_MAX_SIZE=10

# 必须与数据库embedding列的vector(1024)一致
PKA_EMBEDDING_DIMENSION=1024
```

注意：

- `PKA_POSTGRES_*` 供 Docker Compose 创建数据库时使用；
- `PKA_DATABASE_*` 供 Python 应用连接数据库时使用；
- 修改数据库用户名、密码、数据库名或端口时，两组配置必须同步；
- `.env` 包含密钥和密码，不应提交到 Git；
- `.env.example` 只保存示例值，可以提交。

## 8. 数据库结构

初始化 SQL 创建以下扩展：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

主要表：

| 表 | 用途 |
|---|---|
| `pka.database_health_check` | 初始化和健康检查 |
| `pka.documents` | 逻辑文档 |
| `pka.document_versions` | 文档内容版本 |
| `pka.document_chunks` | Chunk、元数据和向量 |
| `pka.user_memories` | 为后续长期记忆预留 |

向量列定义为：

```sql
embedding vector(1024) NOT NULL
```

向量索引使用 HNSW 和余弦距离：

```sql
USING hnsw (embedding vector_cosine_ops)
```

因此 `PKA_EMBEDDING_DIMENSION` 必须为 `1024`。如果更换 Embedding 模型并改变维度，需要同步迁移数据库列和索引。

## 9. PgVectorStore 实现

### 9.1 初始化

`PgVectorStore` 接收：

- `database_url`；
- `schema`；
- `dimension`；
- `pool_min_size`；
- `pool_max_size`；
- `connect_timeout_seconds`；
- `tenant_id`。

构造函数会拒绝：

- 空数据库地址；
- 空 Schema；
- 小于等于 0 的向量维度；
- 非法连接池范围；
- 空 `tenant_id`。

连接池创建时使用 `open=False`，不会在模块导入阶段立即连接数据库。

### 9.2 打开和关闭连接池

```python
await store.open()

try:
    # 执行写入或检索
    ...
finally:
    await store.close()
```

`open()` 会：

1. 创建最小数量的数据库连接；
2. 为每个连接注册 pgvector 类型；
3. 查询 `document_chunks.embedding` 的真实数据库类型；
4. 检查数据库维度是否与应用配置一致。

### 9.3 批量写入

调用形式：

```python
await store.upsert(
    chunks,
    embeddings,
)
```

写入前会完整检查整个批次：

- Chunk 数量必须等于 Embedding 数量；
- 每个向量维度必须正确；
- 向量不能包含 `NaN` 或无穷值；
- 向量不能是全零向量；
- 同一批次不能出现重复 `chunk_id`；
- 同一逻辑文档不能在一个批次中包含多个内容版本。

先验证完整批次，可以避免部分数据已经写入后才发现后续向量非法。

数据库使用 `ON CONFLICT ... DO UPDATE` 实现幂等写入。重复导入相同文档不会无限创建重复 Chunk。

### 9.4 向量检索

调用形式：

```python
results = await store.search(
    query_embedding,
    limit=5,
    filters={
        "document_type": "markdown",
    },
)
```

检索只返回：

- 当前 `tenant_id` 的数据；
- `document_chunks.active = TRUE` 的 Chunk；
- `document_versions.active = TRUE` 的版本。

支持的过滤字段：

- `document_id`；
- `content_hash`；
- `source_path`；
- `file_name`；
- `document_type`；
- `page_number`。

使用未定义的过滤字段会抛出 `UnsupportedVectorFilterError`，过滤字段通过 psycopg 参数传递，不直接拼接用户输入。

检索使用 pgvector 的余弦距离操作符 `<=>`，返回分数为：

```text
score = 1 - cosine_distance
```

最终分数限制在 `[-1, 1]` 范围内。

### 9.5 停用文档

```python
affected = await store.deactivate_document(
    document_id,
)
```

该操作不会物理删除数据，而是：

- 将对应 Chunk 设置为非活动状态；
- 记录停用时间；
- 将对应文档版本设置为非活动状态；
- 返回本次停用的活动 Chunk 数量。

再次停用同一文档会返回 `0`，后续检索不会再返回该文档。

## 10. 工厂接入

`create_vector_store(settings)` 根据配置选择实现：

```text
PKA_VECTOR_STORE_PROVIDER=memory
→ InMemoryVectorStore

PKA_VECTOR_STORE_PROVIDER=pgvector
→ PgVectorStore
```

应用层只依赖统一的 `VectorStoreProvider` 协议，不直接依赖具体数据库。

本地单元测试可以继续使用内存实现，生产和集成测试使用 PostgreSQL 实现。

## 11. FastAPI 生命周期

FastAPI 的 `lifespan` 负责管理数据库连接池：

```text
应用启动
  → 创建ApplicationServices
  → 判断是否为PgVectorStore
  → await vector_store.open()
  → 接收HTTP请求
  → 应用关闭
  → await vector_store.close()
```

这样可以确保：

- 不为每个请求重复建立数据库连接；
- 应用启动时尽早发现数据库或维度配置错误；
- 服务退出时正确释放连接池；
- 注入 Fake Agent Service 的 API 单元测试不要求数据库运行。

## 12. 启动 PostgreSQL

先检查 Compose 配置：

```powershell
docker compose config
```

启动数据库：

```powershell
docker compose up -d postgres
```

查看状态：

```powershell
docker compose ps
```

预期 `pka-postgres` 的状态最终为 `healthy`。

查看日志：

```powershell
docker compose logs postgres --tail 100
```

## 13. 验证数据库基础设施

执行：

```powershell
uv run python scripts/check_database.py
```

应确认：

- PostgreSQL 可以连接；
- PostgreSQL 版本正确；
- pgvector 扩展存在；
- `pka` Schema 存在；
- Day 8 表全部存在；
- `document_chunks.embedding` 为 `vector(1024)`。

也可以直接进入数据库：

```powershell
docker compose exec postgres psql -U pka -d personal_knowledge
```

在 `psql` 中执行：

```sql
SELECT version();
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
\dt pka.*
\d pka.document_chunks
```

退出：

```text
\q
```

## 14. 测试流程

### 14.1 默认测试

```powershell
uv run pytest
```

未设置数据库测试变量时，预期：

```text
369 passed, 3 skipped
```

3 个跳过项是需要真实 PostgreSQL 的集成测试，这是正常行为。

### 14.2 PostgreSQL 集成测试

确保容器为 `healthy`，然后执行：

```powershell
$env:PKA_RUN_POSTGRES_TESTS = "1"
uv run pytest tests\test_pgvector_store.py -vv
```

预期：

```text
3 passed
```

测试覆盖：

1. 批量 upsert 和相似度排序；
2. `document_type` 元数据过滤；
3. 文档停用以及重复停用。

每个测试使用独立的临时 `tenant_id`，结束后删除自己的测试数据。

### 14.3 完整质量检查

```powershell
$env:PKA_RUN_POSTGRES_TESTS = "1"

uv run pytest
uv run ruff check .
uv run mypy
```

当前预期结果：

```text
372 passed
All checks passed!
Success: no issues found
```

结束后清除变量：

```powershell
Remove-Item Env:PKA_RUN_POSTGRES_TESTS -ErrorAction SilentlyContinue
Remove-Item Env:UV_LINK_MODE -ErrorAction SilentlyContinue
```

## 15. 启动 API

数据库为 `healthy` 后执行：

```powershell
uv run uvicorn personal_knowledge_assistant.api.app:app --reload
```

打开：

- Swagger UI：`http://127.0.0.1:8000/docs`；
- 存活检查：`http://127.0.0.1:8000/health/live`；
- 就绪检查：`http://127.0.0.1:8000/health/ready`。

停止 API 使用 `Ctrl+C`。

## 16. 常见问题

### 16.1 `no configuration file provided: not found`

原因：当前目录没有 `compose.yaml`。

检查：

```powershell
Get-Location
Test-Path .\compose.yaml
```

如果 Day 8 文件仍在对应分支，可恢复：

```powershell
git restore --source feature/day8-persistence-foundation -- compose.yaml infra/postgres/init/001_initial_schema.sql scripts/check_database.py
```

### 16.2 Docker Desktop 显示 `Engine stopped`

先检查：

```powershell
wsl --status
wsl -l -v
docker version
```

确认 Docker Desktop 使用 WSL 2 后端，并重新启动 Docker Desktop。WSL 分发处于 `Stopped` 并不一定是故障，Docker 启动时会自动唤醒。

### 16.3 容器一直不是 `healthy`

查看日志：

```powershell
docker compose logs postgres --tail 100
```

重点检查端口占用、数据库密码和数据目录权限。

### 16.4 端口 5432 被占用

在 `.env` 中修改：

```dotenv
PKA_POSTGRES_PORT=5433
PKA_DATABASE_URL=postgresql://pka:pka-local-password@localhost:5433/personal_knowledge
```

然后重新启动 Compose。

### 16.5 修改初始化 SQL 后没有生效

`docker-entrypoint-initdb.d` 只会在空数据卷第一次初始化时执行。

已有数据时应使用数据库迁移。开发阶段如果确定所有本地数据都可删除，才可以执行：

```powershell
docker compose down -v
docker compose up -d postgres
```

`down -v` 会永久删除当前项目的 PostgreSQL 数据卷，不应在存在重要数据时执行。

### 16.6 `VectorDimensionMismatchError`

原因：应用配置的 Embedding 维度与数据库 `vector(1024)` 不一致。

检查：

```powershell
Select-String PKA_EMBEDDING_DIMENSION .env
uv run python scripts/check_database.py
```

当前应统一为 `1024`。

### 16.7 Windows 出现 `ProactorEventLoop` 错误

psycopg 异步模式需要兼容的事件循环。测试通过 `tests/conftest.py` 使用 `SelectorEventLoop`。不要删除该文件。

### 16.8 测试显示 `3 skipped`

这是默认行为。启用真实数据库测试：

```powershell
$env:PKA_RUN_POSTGRES_TESTS = "1"
uv run pytest
```

### 16.9 uv 显示硬链接警告

该提示通常不是失败。PowerShell 中可设置：

```powershell
$env:UV_LINK_MODE = "copy"
```

## 17. 安全注意事项

- 不要把真实密码和 API Key 提交到 Git；
- SQL 表名和 Schema 使用 psycopg Identifier 构造；
- 查询值使用参数绑定；
- 每次查询必须带 `tenant_id`；
- 默认密码只适用于本地开发；
- 生产环境必须更换密码并限制数据库网络访问；
- 不要在有重要数据时执行 `docker compose down -v`。

## 18. Day 9 验收清单

- [x] PostgreSQL + pgvector 容器可以启动；
- [x] 数据库健康检查通过；
- [x] 实现异步 `PgVectorStore`；
- [x] 实现连接池启动和关闭；
- [x] 启动时验证向量维度；
- [x] 实现 Chunk 批量 upsert；
- [x] 实现余弦相似度检索；
- [x] 实现元数据过滤；
- [x] 实现文档逻辑停用；
- [x] 工厂支持 `memory` 和 `pgvector`；
- [x] FastAPI 管理数据库连接池；
- [x] Windows 异步测试兼容；
- [x] PostgreSQL 集成测试通过；
- [x] Ruff、mypy 和完整 pytest 通过。

## 19. 当前限制与下一步

Day 9 完成的是知识 Chunk 的持久化向量检索。尚未实现：

- 文档上传和导入 API；
- 文件更新自动停用旧版本；
- PostgreSQL LangGraph Checkpointer；
- 长期记忆读写接口；
- 用户认证与真实 `tenant_id`；
- 数据库迁移工具；
- 混合检索和 reranker；
- 生产环境密钥管理、备份和监控。

推荐 Day 10 优先实现“文档导入 API + PostgreSQL 增量索引闭环”，让上传文件能够经过解析、切块、Embedding，并最终写入本日完成的 `PgVectorStore`。
