from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy: Mapped[str] = mapped_column(String(50), index=True)
    source: Mapped[str] = mapped_column(String(255))
    page: Mapped[int | None]
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), deferred=True)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    strategy: Mapped[str] = mapped_column(String(50))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100))
    input_tokens: Mapped[int]
    cached_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cost_usd: Mapped[float]
    duration_ms: Mapped[float]
    timings_ms: Mapped[dict[str, float]] = mapped_column(JSONB)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
