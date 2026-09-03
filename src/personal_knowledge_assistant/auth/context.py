from contextvars import (
    ContextVar,
    Token,
)

from personal_knowledge_assistant.auth.models import (
    AuthenticatedPrincipal,
)

_current_principal: ContextVar[AuthenticatedPrincipal | None] = ContextVar(
    "pka_current_principal",
    default=None,
)


def set_current_principal(
    principal: AuthenticatedPrincipal,
) -> Token[AuthenticatedPrincipal | None]:
    """为当前异步请求保存认证身份。"""

    return _current_principal.set(principal)


def reset_current_principal(
    token: Token[AuthenticatedPrincipal | None],
) -> None:
    """请求结束后恢复之前的身份上下文。"""

    _current_principal.reset(token)


def get_current_principal() -> AuthenticatedPrincipal | None:
    """读取当前请求身份。"""

    return _current_principal.get()


def get_current_tenant_id(
    default_tenant_id: str = "local",
) -> str:
    """返回当前租户；非HTTP后台任务使用默认租户。"""

    principal = get_current_principal()

    if principal is None:
        return default_tenant_id

    return principal.tenant_id
