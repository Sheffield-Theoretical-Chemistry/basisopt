from typing import Any


def chunk(x: list[Any], n_chunks: int) -> list[list[Any]]:
    """Chunks an array into roughly equal-sized subarrays."""
    chunk_size = len(x) // n_chunks
    remainder = len(x) % n_chunks
    chunk_list = [chunk_size] * n_chunks
    for i in range(remainder):
        chunk_list[i] += 1
    new_x = []
    ctr = 0
    for i in range(n_chunks):
        new_x.append(x[ctr : ctr + chunk_list[i]])
        ctr += chunk_list[i]
    return new_x
