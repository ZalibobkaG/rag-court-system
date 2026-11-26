import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.logging import setup_logging
from app.db.session import SessionLocal, init_db
from app.services.embeddings import get_embeddings
from app.services.indexing import load_bm25_indexes


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    await init_db()
    await asyncio.to_thread(get_embeddings)  # загрузка FRIDA до первого запроса
    async with SessionLocal() as session:
        await load_bm25_indexes(session)
    yield


app = FastAPI(title="RAG по судебным документам", lifespan=lifespan)
app.include_router(router)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
