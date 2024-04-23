import logging
from typing import Any, Callable, List

import ray

from . import api
from .util import bo_logger

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

from itertools import chain


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


@ray.remote
def process_data(func: Callable[[list[Any]], dict], data_chunk: list[Any], **kwargs):
    """Remote function to process data using the given function."""
    import basisopt as bo

    bo.set_backend('psi4', verbose=False)
    bo.set_tmp_dir('./group_ani_tmp/legendrePairs/mol_group/', verbose=False)
    return func(data_chunk, **kwargs)


def distribute(n_proc: int, func: Callable[[list[Any]], dict], x: list[Any], **kwargs) -> list[Any]:
    """Distributes a function over a desired number of processors using Ray."""
    x = list(x)
    n_chunks = len(x) // n_proc
    if len(x) % n_proc > 0:
        n_chunks += 1
    new_x = chunk(x, n_chunks)

    futures = [process_data.remote(func, chunk, **kwargs) for chunk in new_x]
    results = ray.get(futures)  # Retrieve results
    all_results = list(chain(*results))

    return all_results
