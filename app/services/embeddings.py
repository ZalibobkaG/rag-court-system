from functools import cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


@cache
def get_embeddings() -> HuggingFaceEmbeddings:
    # При первом запуске веса скачиваются с HF Hub в кэш, дальше модель работает локально
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"prompt_name": "search_document"},
        query_encode_kwargs={"prompt_name": "search_query"},
    )
