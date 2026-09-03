import argparse

from personal_knowledge_assistant.auth import (
    JwtAuthenticator,
)
from personal_knowledge_assistant.config import (
    Settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local JWT.")
    parser.add_argument(
        "--tenant-id",
        required=True,
    )
    parser.add_argument(
        "--user-id",
        required=True,
    )
    parser.add_argument(
        "--lifetime-seconds",
        type=int,
        default=3600,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    authenticator = JwtAuthenticator(settings)

    token = authenticator.issue_development_token(
        tenant_id=args.tenant_id,
        user_id=args.user_id,
        lifetime_seconds=(args.lifetime_seconds),
    )

    print(token)


if __name__ == "__main__":
    main()
