from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    update,
)
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


class RequestLog(Base):
    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(32), default=new_id, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    client_requested_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    served_model: Mapped[str] = mapped_column(String(255))
    shadow_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    propensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    stream: Mapped[bool] = mapped_column(Boolean)
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reward_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_posted: Mapped[bool] = mapped_column(Boolean, default=False)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "client_requested_model": self.client_requested_model,
            "served_model": self.served_model,
            "shadow_model": self.shadow_model,
            "decision_id": self.decision_id,
            "propensity": self.propensity,
            "stream": self.stream,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "cost_source": self.cost_source,
            "error": self.error,
            "quality": self.quality,
            "quality_source": self.quality_source,
            "reward_value": self.reward_value,
            "reward_posted": self.reward_posted,
        }


def make_session_factory(database_url: str) -> sessionmaker:
    database_url = normalize_database_url(database_url)
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def claim_reward_slot(session_factory: sessionmaker, request_id: str) -> bool:
    """Atomically claim the right to post THE one reward for a request.

    Explicit feedback and the background judge can race under real latency;
    the single UPDATE ... WHERE reward_posted = false guarantees exactly one
    winner regardless of interleaving.
    """
    with session_factory() as session:
        result = session.execute(
            update(RequestLog)
            .where(
                RequestLog.request_id == request_id,
                RequestLog.reward_posted.is_(False),
            )
            .values(reward_posted=True)
        )
        session.commit()
        return int(result.rowcount or 0) == 1


def release_reward_slot(session_factory: sessionmaker, request_id: str) -> None:
    """Reopen the slot after a failed post so a later reward can still land."""
    with session_factory() as session:
        session.execute(
            update(RequestLog)
            .where(RequestLog.request_id == request_id)
            .values(reward_posted=False)
        )
        session.commit()


def prune_history(session_factory: sessionmaker, cutoff: datetime) -> int:
    with session_factory() as session:
        result = session.execute(
            delete(RequestLog).where(RequestLog.created_at < cutoff)
        )
        session.commit()
        return int(result.rowcount or 0)
