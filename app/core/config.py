from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://rag:rag@db:5432/rag"
    docs_dir: Path = Path("/data")
    log_level: str = "INFO"

    # Эмбеддинги: FRIDA (SaluteDevices / Сбер), считается локально
    embedding_model: str = "ai-forever/FRIDA"
    embedding_dim: int = 1536

    # Рекурсивный чанкинг, длина в токенах FRIDA (лимит модели 512)
    chunk_size: int = 400
    chunk_overlap: int = 60

    # Гибридный поиск
    vector_k: int = 20
    bm25_k: int = 20
    top_k: int = 10
    rrf_k: int = 60

    # LLM: DeepSeek
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_thinking: Literal["enabled", "disabled"] = "disabled"

    # Пиковые цены, $ за 1 млн токенов; вне пика вдвое ниже
    price_input_cache_hit: float = 0.006
    price_input_cache_miss: float = 0.3
    price_output: float = 1.2


settings = Settings()
