from pydantic import BaseModel


class Candidate(BaseModel):
    chunk_id: int
    source: str
    page: int | None
    text: str
    vector_rank: int | None = None
    vector_distance: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    top_rank: int | None = None  # номер [n] в промпте, если чанк попал в топ
    cited: bool = False  # LLM сослалась на чанк в ответе


class Usage(BaseModel):
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float


class IndexRequest(BaseModel):
    strategy: str = "recursive"


class IndexResponse(BaseModel):
    strategy: str
    files: int
    chunks: int
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str
    strategy: str = "recursive"


class AskResponse(BaseModel):
    answer: str
    usage: Usage
    timings_ms: dict[str, float]
    candidates: list[Candidate]


class StatsResponse(BaseModel):
    queries: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    avg_duration_ms: float
