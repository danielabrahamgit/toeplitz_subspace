from collections import OrderedDict
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

# https://stackoverflow.com/questions/54076972/returning-value-when-exiting-python-context-manager
class Timer():
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, name: str, verbose=False):
        self.name = name
        self.start = None
        self.total = None
        self.verbose = verbose

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, type=None, value=None, traceback=None):
        self.total = (time.perf_counter() - self.start) # Store the desired value.
        if self.verbose:
            print(f'{self.name}: {self.total:0.5f} s')
