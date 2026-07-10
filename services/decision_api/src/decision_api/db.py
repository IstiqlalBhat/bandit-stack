"""SQLAlchemy models. SQLite by default, Postgres via DATABASE URL — the
schema keeps JSON columns portable across both."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")
TIMESTAMP = DateTime(timezone=True)


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    arms: Mapped[list] = mapped_column(JSON_DOCUMENT)
    policy_config: Mapped[dict] = mapped_column(JSON_DOCUMENT)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)


class DecisionRecord(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.id"), index=True)
    context: Mapped[dict | None] = mapped_column(JSON_DOCUMENT, nullable=True)
    arm_index: Mapped[int] = mapped_column(Integer)
    arm_name: Mapped[str] = mapped_column(String(255))
    propensity: Mapped[float] = mapped_column(Float)
    policy_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)


class RewardRecord(Base):
    __tablename__ = "rewards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), index=True)
    component: Mapped[str] = mapped_column(String(64), default="explicit")
    value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)


class PolicySnapshot(Base):
    __tablename__ = "policy_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    state: Mapped[dict] = mapped_column(JSON_DOCUMENT)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=utcnow)


def make_session_factory(database_url: str) -> sessionmaker:
    database_url = normalize_database_url(database_url)
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
