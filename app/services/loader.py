from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loguru import logger

LOADERS: dict[str, type[BaseLoader]] = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}


async def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(folder.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            if not path.name.startswith("."):
                logger.warning("Пропущен файл неподдерживаемого формата: {}", path.name)
            continue
        for doc in await loader(str(path)).aload():
            page = doc.metadata.get("page")
            doc.metadata = {"source": path.name, "page": page + 1 if page is not None else None}
            documents.append(doc)
    return documents
