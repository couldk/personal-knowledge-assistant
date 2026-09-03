from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from personal_knowledge_assistant.auth import (
    AuthenticatedPrincipal,
    AuthenticationError,
    AuthenticationMiddleware,
    JwtAuthenticator,
    get_current_principal,
    get_current_tenant_id,
    reset_current_principal,
    set_current_principal,
)
from personal_knowledge_assistant.config import Settings

TEST_SECRET = "day12-test-secret-key-must-be-at-least-32-characters"


def make_auth_settings(
    *,
    enabled: bool = True,
) -> Settings:
    """创建不依赖本地.env的认证测试配置。"""

    return Settings(
        auth_enabled=enabled,
        auth_jwt_secret_key=(SecretStr(TEST_SECRET) if enabled else None),
        auth_jwt_algorithm="HS256",
        auth_jwt_issuer=("personal-knowledge-assistant"),
        auth_jwt_audience=("personal-knowledge-assistant-api"),
        auth_local_tenant_id="local",
        auth_local_user_id="local-user",
    )


def create_auth_test_app(
    settings: Settings,
) -> FastAPI:
    """创建只用于测试认证中间件的应用。"""

    app = FastAPI()

    app.add_middleware(
        AuthenticationMiddleware,
        authenticator=JwtAuthenticator(
            settings,
        ),
    )

    @app.get("/protected")
    async def protected(
        request: Request,
    ) -> dict[str, str]:
        principal = request.state.principal
        context_principal = get_current_principal()

        assert isinstance(
            principal,
            AuthenticatedPrincipal,
        )
        assert context_principal is not None

        return {
            "tenant_id": principal.tenant_id,
            "user_id": principal.user_id,
            "context_tenant_id": (context_principal.tenant_id),
        }

    @app.get("/health/live")
    async def public_health() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


def test_disabled_authentication_uses_local_principal() -> None:
    settings = make_auth_settings(
        enabled=False,
    )
    authenticator = JwtAuthenticator(
        settings,
    )

    principal = authenticator.authenticate(
        None,
    )

    assert principal.tenant_id == "local"
    assert principal.user_id == "local-user"


def test_issue_and_authenticate_token() -> None:
    settings = make_auth_settings()
    authenticator = JwtAuthenticator(
        settings,
    )

    token = authenticator.issue_development_token(
        tenant_id="tenant-a",
        user_id="user-a",
    )

    principal = authenticator.authenticate(
        f"Bearer {token}",
    )

    assert principal == AuthenticatedPrincipal(
        tenant_id="tenant-a",
        user_id="user-a",
    )


def test_authenticator_rejects_missing_header() -> None:
    authenticator = JwtAuthenticator(
        make_auth_settings(),
    )

    with pytest.raises(
        AuthenticationError,
        match="Authorization header is required",
    ):
        authenticator.authenticate(None)


def test_protected_endpoint_rejects_missing_token() -> None:
    app = create_auth_test_app(
        make_auth_settings(),
    )

    with TestClient(app) as client:
        response = client.get(
            "/protected",
        )

    assert response.status_code == 401
    assert response.json() == {"detail": ("Authentication credentials are invalid or missing.")}
    assert response.headers["www-authenticate"] == "Bearer"


def test_protected_endpoint_accepts_valid_token() -> None:
    settings = make_auth_settings()
    authenticator = JwtAuthenticator(
        settings,
    )
    token = authenticator.issue_development_token(
        tenant_id="tenant-a",
        user_id="user-a",
    )
    app = create_auth_test_app(
        settings,
    )

    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={
                "Authorization": (f"Bearer {token}"),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "context_tenant_id": "tenant-a",
    }

    # 请求完成后必须清除ContextVar，
    # 避免身份泄漏到下一个请求。
    assert get_current_principal() is None


def test_protected_endpoint_rejects_invalid_token() -> None:
    app = create_auth_test_app(
        make_auth_settings(),
    )

    with TestClient(app) as client:
        response = client.get(
            "/protected",
            headers={
                "Authorization": ("Bearer invalid-token"),
            },
        )

    assert response.status_code == 401


def test_public_endpoint_does_not_require_token() -> None:
    app = create_auth_test_app(
        make_auth_settings(),
    )

    with TestClient(app) as client:
        response = client.get(
            "/health/live",
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_tenant_context_uses_current_principal() -> None:
    principal = AuthenticatedPrincipal(
        tenant_id="tenant-b",
        user_id="user-b",
    )

    token = set_current_principal(
        principal,
    )

    try:
        assert get_current_tenant_id() == "tenant-b"
        assert get_current_principal() == principal
    finally:
        reset_current_principal(token)

    assert get_current_principal() is None
    assert get_current_tenant_id("fallback") == "fallback"
