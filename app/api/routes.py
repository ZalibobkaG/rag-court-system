from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.repositories.queries import get_stats
from app.schemas import AskRequest, AskResponse, IndexRequest, IndexResponse, StatsResponse
from app.services.bm25 import bm25_store
from app.services.chunking import CHUNKERS
from app.services.indexing import index_documents
from app.services.rag import ask

router = APIRouter(prefix="/api")
Session = Annotated[AsyncSession, Depends(get_session)]


def _check_strategy(strategy: str) -> None:
    if strategy not in CHUNKERS:
        raise HTTPException(404, f"Неизвестная стратегия чанкинга: {strategy}")


@router.get("/strategies")
async def strategies() -> list[str]:
    return list(CHUNKERS)


@router.post("/index")
async def index(body: IndexRequest, session: Session) -> IndexResponse:
    _check_strategy(body.strategy)
    return await index_documents(session, body.strategy)


@router.post("/ask")
async def ask_question(body: AskRequest, session: Session) -> AskResponse:
    _check_strategy(body.strategy)
    if body.strategy not in bm25_store:
        raise HTTPException(409, f"Документы ещё не проиндексированы стратегией {body.strategy}")
    return await ask(session, body.strategy, body.question)


@router.get("/stats")
async def stats(session: Session) -> StatsResponse:
    return await get_stats(session)
