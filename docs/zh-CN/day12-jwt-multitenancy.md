# Day 12：JWT 认证与请求级多租户隔离

## 完成内容

- 使用 HS256 JWT 验证 API 请求身份。
- 从 JWT 的 `sub` 和 `tenant_id` 声明构造当前用户身份。
- 使用 `ContextVar` 在单个异步请求内传递身份。
- PostgreSQL 文档和向量操作自动使用当前请求的 `tenant_id`。
- 认证关闭时保留本地开发身份，生产环境强制开启认证。
- 健康检查和 API 文档保持公开，其余 HTTP 路径要求 Bearer Token。

## 本地配置

复制 `.env.example` 中的认证配置到 `.env`：

```dotenv
PKA_AUTH_ENABLED=true
PKA_AUTH_JWT_SECRET_KEY=请替换为至少32个字符的随机密钥
PKA_AUTH_JWT_ALGORITHM=HS256
PKA_AUTH_JWT_ISSUER=personal-knowledge-assistant
PKA_AUTH_JWT_AUDIENCE=personal-knowledge-assistant-api
PKA_AUTH_LOCAL_TENANT_ID=local
PKA_AUTH_LOCAL_USER_ID=local-user
```

不要提交包含真实密钥的 `.env`。

## 创建开发 Token

```powershell
uv run python scripts/create_dev_token.py --tenant-id tenant-a --user-id user-a
```

把输出保存到当前 PowerShell 会话：

```powershell
$token = uv run python scripts/create_dev_token.py --tenant-id tenant-a --user-id user-a
```

## 启动 API

Windows 开发环境使用 `--reload`，以便 Psycopg 使用兼容的 Selector 事件循环：

```powershell
uv run uvicorn personal_knowledge_assistant.api:create_api_app --factory --host 127.0.0.1 --port 8000 --reload
```

## 验证认证

无 Token 的受保护请求应返回 `401`：

```powershell
curl.exe -i http://127.0.0.1:8000/api/v1/documents
```

携带 Token 后应通过认证：

```powershell
curl.exe -i http://127.0.0.1:8000/api/v1/documents -H "Authorization: Bearer $token"
```

健康检查不要求 Token：

```powershell
curl.exe http://127.0.0.1:8000/health/live
```

## 多租户行为

`tenant-a` 的 Token 只会在 PostgreSQL 查询中使用 `tenant-a`。使用 `tenant-b` 的 Token 发起相同请求时，查询会自动切换为 `tenant-b`，从而隔离文档、版本和向量数据。

后台任务或认证关闭的本地开发调用使用 `PKA_AUTH_LOCAL_TENANT_ID`。

## 验收命令

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

PostgreSQL 容器运行时，可以额外执行集成测试：

```powershell
$env:PKA_RUN_POSTGRES_TESTS = "1"
uv run pytest tests/test_pgvector_store.py -v
```
