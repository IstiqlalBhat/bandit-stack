from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from decision_api.db import Route, normalize_database_url


def test_bare_postgresql_url_uses_psycopg_driver():
    assert (
        normalize_database_url("postgresql://user:pass@db.example:5432/app?sslmode=require")
        == "postgresql+psycopg://user:pass@db.example:5432/app?sslmode=require"
    )


def test_legacy_postgres_url_uses_psycopg_driver():
    assert normalize_database_url("postgres://user:pass@db/app") == (
        "postgresql+psycopg://user:pass@db/app"
    )


def test_postgresql_schema_uses_jsonb_and_timezone_aware_timestamps():
    ddl = str(CreateTable(Route.__table__).compile(dialect=postgresql.dialect()))
    assert "arms JSONB" in ddl
    assert "created_at TIMESTAMP WITH TIME ZONE" in ddl
