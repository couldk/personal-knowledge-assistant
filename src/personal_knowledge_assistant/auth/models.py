from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

IdentityText = Field(
    min_length=1,
    max_length=128,
    pattern=r"^[A-Za-z0-9._:-]+$",
)


class AuthenticatedPrincipal(BaseModel):
    """已经通过认证的租户用户身份。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tenant_id: str = IdentityText
    user_id: str = IdentityText
