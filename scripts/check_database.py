from __future__ import annotations

import sys
from typing import Any

import psycopg
from psycopg import sql

from personal_knowledge_assistant.config import Settings

REQUIRED_TABLES = frozenset(
    {
        "documents",
        "document_versions",
        "document_chunks",
        "user_memories",
    }
)


def fetch_table_names(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
) -> set[str]:
    """读取指定Schema中的业务表名称。"""

    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        """,
        (schema,),
    ).fetchall()

    return {str(row[0]) for row in rows}


def fetch_vector_extension_version(
    connection: psycopg.Connection[Any],
) -> str:
    """返回数据库中的pgvector扩展版本。"""

    row = connection.execute(
        """
        SELECT extversion
        FROM pg_extension
        WHERE extname = 'vector'
        """
    ).fetchone()

    if row is None:
        raise RuntimeError("The pgvector extension is not installed.")

    return str(row[0])


def verify_schema_exists(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
) -> None:
    """确认应用Schema已经创建。"""

    row = connection.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = %s
        )
        """,
        (schema,),
    ).fetchone()

    if row is None or not bool(row[0]):
        raise RuntimeError(f"Database schema does not exist: {schema}")


def fetch_chunk_count(
    connection: psycopg.Connection[Any],
    *,
    schema: str,
) -> int:
    """返回活动文档片段数量。"""

    query = sql.SQL(
        """
        SELECT COUNT(*)
        FROM {}.document_chunks
        WHERE active = TRUE
        """
    ).format(sql.Identifier(schema))

    row = connection.execute(query).fetchone()

    if row is None:
        return 0

    return int(row[0])


def verify_database(settings: Settings) -> None:
    """验证数据库连接、扩展、Schema和业务表。"""

    with psycopg.connect(
        settings.database_url,
        connect_timeout=(settings.database_connect_timeout_seconds),
        autocommit=True,
    ) as connection:
        metadata_row = connection.execute(
            """
            SELECT
                current_database(),
                current_user,
                current_setting('server_version')
            """
        ).fetchone()

        if metadata_row is None:
            raise RuntimeError("Database metadata query returned no result.")

        vector_version = fetch_vector_extension_version(connection)

        verify_schema_exists(
            connection,
            schema=settings.database_schema,
        )

        table_names = fetch_table_names(
            connection,
            schema=settings.database_schema,
        )

        missing_tables = REQUIRED_TABLES - table_names

        if missing_tables:
            names = ", ".join(sorted(missing_tables))
            raise RuntimeError(f"Database tables are missing: {names}")

        chunk_count = fetch_chunk_count(
            connection,
            schema=settings.database_schema,
        )

        print("Database connection: OK")
        print(f"Database: {metadata_row[0]}")
        print(f"User: {metadata_row[1]}")
        print(f"PostgreSQL: {metadata_row[2]}")
        print(f"pgvector: {vector_version}")
        print(f"Schema: {settings.database_schema}")
        print("Tables: " + ", ".join(sorted(table_names)))
        print(f"Active chunks: {chunk_count}")


def main() -> int:
    """运行数据库验证并返回进程退出码。"""

    try:
        verify_database(Settings())
    except Exception as exc:
        print(
            f"Database verification failed: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
