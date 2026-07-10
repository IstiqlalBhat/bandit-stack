from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from llm_proxy.db import RequestLog, normalize_database_url


def test_postgres_urls_use_psycopg_without_changing_explicit_drivers():
    assert normalize_database_url("postgres://user:pass@db/proxy") == (
        "postgresql+psycopg://user:pass@db/proxy"
    )
    assert normalize_database_url("postgresql://user:pass@db/proxy") == (
        "postgresql+psycopg://user:pass@db/proxy"
    )
    assert normalize_database_url("postgresql+asyncpg://user:pass@db/proxy") == (
        "postgresql+asyncpg://user:pass@db/proxy"
    )


def test_postgresql_schema_uses_timezone_aware_timestamps():
    ddl = str(CreateTable(RequestLog.__table__).compile(dialect=postgresql.dialect()))
    assert "created_at TIMESTAMP WITH TIME ZONE" in ddl
