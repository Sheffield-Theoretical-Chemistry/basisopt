from typing import Any, Callable, List
import logging
from . import api
from .util import bo_logger


if api._PARALLEL:
    from distributed import Client, LocalCluster, wait, WorkerPlugin
    import dask
else:
    bo_logger.warning("Dask not installed, parallelisation not available")

from itertools import chain
def chunk(x: list[Any], n_chunks: int) -> list[list[Any]]:
    """Chunks an array into roughly equal-sized subarrays

    Args:
         x - array of values of length L
         n_chunks - number of chunks to split into

    Returns:
         a list of n_chunks arrays of length L//n_chunks
         or (L//n_chunks)+1
    """
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

class InitializeBackend(WorkerPlugin):
    def __init__(self, backend):
        self.backend = backend

    def setup(self, worker=None):
        import basisopt as bo
        bo.set_backend(self.backend,verbose=False)
        bo.set_tmp_dir('./scr/',verbose=False)

def distribute(n_proc: int, func: Callable[[list[Any]], dict], x: list[Any], **kwargs
) -> list[Any]:
    """Distributes a function over a desired no. of procs
    using the distributed library.

    Args:
         n_proc - the number of processes to start
         func - the function to call, with signature (x, **kwargs)
         x  - the array of values to distribute over
         kwargs - the named arguments accepted by func
    Returns:
         a list of results ordered by process ID
    """

    # Create cluster and client once

    dask.config.set({"global-config": "./dask.yaml"})
    cluster = LocalCluster(n_workers=n_proc, processes=True, silence_logs=logging.ERROR)
    client = Client(cluster)

    # Register the worker plugin to set the backend
    plugin = InitializeBackend('psi4')
    client.register_worker_plugin(plugin)

    # Split data into chunks
    x = list(x)
    n_chunks = len(x) // n_proc
    if len(x) % n_proc > 0:
        n_chunks += 1
    new_x = chunk(x, n_chunks)

    all_results = []
    for chunk_i in new_x:
        # Submit tasks
        futures = client.map(func, chunk_i, **kwargs)

        # Wait for completion
        wait(futures)

        # Retrieve results
        results = client.gather(futures)
        all_results.extend(results)

    # Close the client and cluster
    client.close()
    cluster.close()
    #print(all_results[0])
    return all_results
