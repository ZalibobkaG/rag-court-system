from datetime import UTC, datetime
from functools import cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from app.core.config import settings
from app.schemas import Usage

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты помощник юриста. Отвечай на вопрос только по приведённым фрагментам судебных документов. "
            "После каждого утверждения указывай номер фрагмента-источника в квадратных скобках, например [3]. "
            "Если во фрагментах нет ответа, так и скажи.",
        ),
        ("human", "Фрагменты:\n\n{context}\n\nВопрос: {question}"),
    ]
)


@cache
def get_llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        extra_body={"thinking": {"type": settings.llm_thinking}},
    )


def is_peak(moment: datetime) -> bool:
    """Пиковые часы DeepSeek: 01–04 и 06–10 UTC, пн–пт. Китайские праздники не учитываются."""
    return moment.weekday() < 5 and (1 <= moment.hour < 4 or 6 <= moment.hour < 10)


def cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int, moment: datetime) -> float:
    cost = (
        (input_tokens - cached_tokens) * settings.price_input_cache_miss
        + cached_tokens * settings.price_input_cache_hit
        + output_tokens * settings.price_output
    ) / 1_000_000
    return round(cost if is_peak(moment) else cost / 2, 6)


async def generate(context: str, question: str) -> tuple[str, Usage]:
    moment = datetime.now(UTC)
    message = await (PROMPT | get_llm()).ainvoke({"context": context, "question": question})
    usage = message.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
    output_tokens = usage.get("output_tokens", 0)
    return message.text, Usage(
        model=settings.llm_model,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(input_tokens, cached_tokens, output_tokens, moment),
    )
