from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Query
from app.schemas import StatsResponse


async def add_query(session: AsyncSession, query: Query) -> None:
    session.add(query)
    await session.commit()


async def get_stats(session: AsyncSession) -> StatsResponse:
    row = (
        await session.execute(
            select(
                func.count(Query.id),
                func.coalesce(func.sum(Query.input_tokens), 0),
                func.coalesce(func.sum(Query.output_tokens), 0),
                func.coalesce(func.sum(Query.cost_usd), 0.0),
                func.coalesce(func.avg(Query.duration_ms), 0.0),
            )
        )
    ).one()
    return StatsResponse(
        queries=row[0], input_tokens=row[1], output_tokens=row[2], cost_usd=row[3], avg_duration_ms=row[4]
    )
