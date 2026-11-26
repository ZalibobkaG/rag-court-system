import re

import snowballstemmer
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core.config import settings

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    # Стеммер создаётся на каждый вызов: объект snowballstemmer не потокобезопасен
    words = _WORD.findall(text.lower().replace("ё", "е"))
    return snowballstemmer.stemmer("russian").stemWords(words)


class BM25Store:
    """BM25-индексы в памяти, по одному на стратегию чанкинга."""

    def __init__(self) -> None:
        self._retrievers: dict[str, BM25Retriever] = {}

    def build(self, strategy: str, documents: list[Document]) -> None:
        self._retrievers[strategy] = BM25Retriever.from_documents(
            documents, k=settings.bm25_k, preprocess_func=tokenize
        )

    def __contains__(self, strategy: str) -> bool:
        return strategy in self._retrievers

    async def search(self, strategy: str, query: str) -> list[Document]:
        retriever = self._retrievers.get(strategy)
        return await retriever.ainvoke(query) if retriever else []


bm25_store = BM25Store()
