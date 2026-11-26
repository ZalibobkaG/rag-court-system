import asyncio

from langchain_core.documents import Document
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk
from app.repositories.chunks import list_chunks, replace_chunks
from app.schemas import IndexResponse
from app.services.bm25 import bm25_store
from app.services.chunking import CHUNKERS
from app.services.embeddings import get_embeddings
from app.services.loader import load_documents
from app.services.timing import timer


def _to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text, metadata={"chunk_id": chunk.id, "source": chunk.source, "page": chunk.page}
    )


async def index_documents(session: AsyncSession, strategy: str) -> IndexResponse:
    timings: dict[str, float] = {}
    with timer(timings, "load_ms"):
        documents = await load_documents(settings.docs_dir)
    with timer(timings, "chunk_ms"):
        pieces = await asyncio.to_thread(CHUNKERS[strategy], documents)
    with timer(timings, "embed_ms"):
        vectors = await get_embeddings().aembed_documents([piece.page_content for piece in pieces])
    chunks = [
        Chunk(strategy=strategy, source=piece.metadata["source"], page=piece.metadata["page"],
              text=piece.page_content, embedding=vector)
        for piece, vector in zip(pieces, vectors, strict=True)
    ]
    with timer(timings, "save_ms"):
        await replace_chunks(session, strategy, chunks)
    if chunks:
        bm25_store.build(strategy, [_to_document(chunk) for chunk in chunks])

    files = len({doc.metadata["source"] for doc in documents})
    logger.info("Индексация [{}]: файлов {}, чанков {}, время {}", strategy, files, len(chunks), timings)
    return IndexResponse(strategy=strategy, files=files, chunks=len(chunks), timings_ms=timings)


async def load_bm25_indexes(session: AsyncSession) -> None:
    for strategy in CHUNKERS:
        if chunks := await list_chunks(session, strategy):
            bm25_store.build(strategy, [_to_document(chunk) for chunk in chunks])
            logger.info("BM25 [{}]: загружено чанков {}", strategy, len(chunks))
