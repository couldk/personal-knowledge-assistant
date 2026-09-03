from datetime import (
    UTC,
    datetime,
    timedelta,
)
from typing import Any

import jwt
from pydantic import ValidationError

from personal_knowledge_assistant.auth.models import (
    AuthenticatedPrincipal,
)
from personal_knowledge_assistant.config import Settings


class AuthenticationError(Exception):
    """JWT认证失败。"""


class JwtAuthenticator:
    """验证和签发本地开发JWT。"""

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._enabled = settings.auth_enabled
        self._algorithm = settings.auth_jwt_algorithm
        self._issuer = settings.auth_jwt_issuer
        self._audience = settings.auth_jwt_audience
        self._local_principal = AuthenticatedPrincipal(
            tenant_id=(settings.auth_local_tenant_id),
            user_id=(settings.auth_local_user_id),
        )

        secret = settings.auth_jwt_secret_key

        self._secret = secret.get_secret_value() if secret is not None else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def local_principal(
        self,
    ) -> AuthenticatedPrincipal:
        return self._local_principal

    def authenticate(
        self,
        authorization: str | None,
    ) -> AuthenticatedPrincipal:
        """验证Authorization Bearer头。"""

        if not self._enabled:
            return self._local_principal

        if authorization is None:
            raise AuthenticationError("Authorization header is required.")

        scheme, separator, token = authorization.partition(" ")

        if not separator or scheme.casefold() != "bearer" or not token.strip():
            raise AuthenticationError("Authorization header must use Bearer.")

        if self._secret is None:
            raise AuthenticationError("JWT authentication is not configured.")

        try:
            payload: dict[str, Any] = jwt.decode(
                token.strip(),
                self._secret,
                algorithms=[self._algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": [
                        "exp",
                        "iat",
                        "iss",
                        "aud",
                        "sub",
                        "tenant_id",
                    ]
                },
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("JWT is invalid or expired.") from exc

        try:
            return AuthenticatedPrincipal(
                tenant_id=str(payload["tenant_id"]),
                user_id=str(payload["sub"]),
            )
        except (
            KeyError,
            TypeError,
            ValidationError,
        ) as exc:
            raise AuthenticationError("JWT identity claims are invalid.") from exc

    def issue_development_token(
        self,
        *,
        tenant_id: str,
        user_id: str,
        lifetime_seconds: int = 3600,
    ) -> str:
        """生成本地开发测试Token。"""

        if self._secret is None:
            raise AuthenticationError("JWT secret is not configured.")

        if lifetime_seconds < 60:
            raise ValueError("Token lifetime must be at least 60 seconds.")

        principal = AuthenticatedPrincipal(
            tenant_id=tenant_id,
            user_id=user_id,
        )
        now = datetime.now(tz=UTC)

        return jwt.encode(
            {
                "iss": self._issuer,
                "aud": self._audience,
                "sub": principal.user_id,
                "tenant_id": principal.tenant_id,
                "iat": now,
                "exp": now + timedelta(seconds=lifetime_seconds),
            },
            self._secret,
            algorithm=self._algorithm,
        )
