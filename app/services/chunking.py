from collections.abc import Callable
from functools import cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.core.config import settings

Chunker = Callable[[list[Document]], list[Document]]


@cache
def _tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(settings.embedding_model)


def recursive_chunker(documents: list[Document]) -> list[Document]:
    """Режет по абзацам, затем по строкам, предложениям и словам, пока чанк не влезет в chunk_size токенов."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return splitter.split_documents(documents)


# Новые методы чанкинга добавляются сюда и сравниваются по имени стратегии
CHUNKERS: dict[str, Chunker] = {"recursive": recursive_chunker}
