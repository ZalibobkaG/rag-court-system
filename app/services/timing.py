from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(timings: dict[str, float], name: str) -> Generator[None, None, None]:
    start = perf_counter()
    yield
    timings[name] = round((perf_counter() - start) * 1000, 1)
