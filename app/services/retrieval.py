from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.repositories.chunks import search_chunks
from app.schemas import Candidate
from app.services.bm25 import bm25_store
from app.services.embeddings import get_embeddings
from app.services.timing import timer


async def hybrid_search(
    session: AsyncSession, strategy: str, question: str, timings: dict[str, float]
) -> list[Candidate]:
    """Векторный поиск + BM25, слияние через Reciprocal Rank Fusion. Возвращает всех кандидатов по убыванию RRF."""
    with timer(timings, "embed_query_ms"):
        query_vector = await get_embeddings().aembed_query(question)
    with timer(timings, "vector_ms"):
        vector_hits = await search_chunks(session, strategy, query_vector, settings.vector_k)
    with timer(timings, "bm25_ms"):
        bm25_hits = await bm25_store.search(strategy, question)

    candidates: dict[int, Candidate] = {}
    for rank, (chunk, distance) in enumerate(vector_hits, 1):
        candidate = candidates.setdefault(
            chunk.id, Candidate(chunk_id=chunk.id, source=chunk.source, page=chunk.page, text=chunk.text)
        )
        candidate.vector_rank, candidate.vector_distance = rank, round(distance, 4)
        candidate.rrf_score += 1 / (settings.rrf_k + rank)
    for rank, doc in enumerate(bm25_hits, 1):
        candidate = candidates.setdefault(
            doc.metadata["chunk_id"], Candidate(text=doc.page_content, **doc.metadata)
        )
        candidate.bm25_rank = rank
        candidate.rrf_score += 1 / (settings.rrf_k + rank)

    ranked = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
    for position, candidate in enumerate(ranked[: settings.top_k], 1):
        candidate.top_rank = position
    return ranked
