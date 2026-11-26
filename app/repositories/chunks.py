from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk


async def replace_chunks(session: AsyncSession, strategy: str, chunks: list[Chunk]) -> None:
    await session.execute(delete(Chunk).where(Chunk.strategy == strategy))
    session.add_all(chunks)
    await session.commit()


async def list_chunks(session: AsyncSession, strategy: str) -> Sequence[Chunk]:
    result = await session.scalars(select(Chunk).where(Chunk.strategy == strategy).order_by(Chunk.id))
    return result.all()


async def search_chunks(
    session: AsyncSession, strategy: str, embedding: list[float], limit: int
) -> list[tuple[Chunk, float]]:
    distance = Chunk.embedding.cosine_distance(embedding).label("distance")
    result = await session.execute(
        select(Chunk, distance).where(Chunk.strategy == strategy).order_by(distance).limit(limit)
    )
    return [(chunk, dist) for chunk, dist in result.tuples()]
