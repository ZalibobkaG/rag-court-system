import re

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Query
from app.repositories.queries import add_query
from app.schemas import AskResponse, Candidate
from app.services.llm import generate
from app.services.retrieval import hybrid_search
from app.services.timing import timer

_CITATION = re.compile(r"\[([\d,\s]+)\]")


def _format_context(top: list[Candidate]) -> str:
    return "\n\n".join(
        f"[{c.top_rank}] {c.source}{f', стр. {c.page}' if c.page else ''}\n{c.text}" for c in top
    )


def _cited_numbers(answer: str) -> set[int]:
    return {int(n) for group in _CITATION.findall(answer) for n in group.split(",") if n.strip()}


async def ask(session: AsyncSession, strategy: str, question: str) -> AskResponse:
    timings: dict[str, float] = {}
    with timer(timings, "total_ms"):
        candidates = await hybrid_search(session, strategy, question, timings)
        top = [c for c in candidates if c.top_rank]
        with timer(timings, "llm_ms"):
            answer, usage = await generate(_format_context(top), question)

    cited = _cited_numbers(answer)
    for candidate in top:
        candidate.cited = candidate.top_rank in cited

    await add_query(
        session,
        Query(
            strategy=strategy, question=question, answer=answer, model=usage.model,
            input_tokens=usage.input_tokens, cached_tokens=usage.cached_tokens, output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd, duration_ms=timings["total_ms"], timings_ms=timings,
            candidates=[c.model_dump(exclude={"text"}) for c in candidates],
        ),
    )
    logger.info(
        "Вопрос [{}]: кандидатов {}, в топе {}, процитировано {}, токены {}/{} (кэш {}), ${}, время {}",
        strategy, len(candidates), len(top), sorted(cited), usage.input_tokens, usage.output_tokens,
        usage.cached_tokens, usage.cost_usd, timings,
    )
    return AskResponse(answer=answer, usage=usage, timings_ms=timings, candidates=candidates)
