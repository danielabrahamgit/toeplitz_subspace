import contextlib
import time
from typing import Optional

@contextlib.contextmanager
def tictoc(name: Optional[str] = None, verbose: bool = False):
    if verbose:
        print(f'> Running {name}...')
    start = time.perf_counter()
    yield
    total = time.perf_counter() - start
    if verbose:
        print(f'>> Time: {total} s')
