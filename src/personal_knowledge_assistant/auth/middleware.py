from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import (
    ASGIApp,
    Receive,
    Scope,
    Send,
)

from personal_knowledge_assistant.auth.context import (
    reset_current_principal,
    set_current_principal,
)
from personal_knowledge_assistant.auth.jwt_auth import (
    AuthenticationError,
    JwtAuthenticator,
)


class AuthenticationMiddleware:
    """对受保护的HTTP接口执行JWT认证。"""

    _PUBLIC_PATHS = frozenset(
        {
            "/health/live",
            "/health/ready",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
            "/openapi.json",
        }
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        authenticator: JwtAuthenticator,
    ) -> None:
        self._app = app
        self._authenticator = authenticator

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(
                scope,
                receive,
                send,
            )
            return

        path = str(scope.get("path", ""))

        if path in self._PUBLIC_PATHS:
            await self._app(
                scope,
                receive,
                send,
            )
            return

        headers = Headers(scope=scope)

        try:
            principal = self._authenticator.authenticate(headers.get("authorization"))
        except AuthenticationError:
            response = JSONResponse(
                status_code=401,
                content={"detail": ("Authentication credentials are invalid or missing.")},
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

            await response(
                scope,
                receive,
                send,
            )
            return

        scope.setdefault(
            "state",
            {},
        )["principal"] = principal

        token = set_current_principal(principal)

        try:
            await self._app(
                scope,
                receive,
                send,
            )
        finally:
            reset_current_principal(token)
