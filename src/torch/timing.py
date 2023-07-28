from collections import OrderedDict
import contextlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

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
class Timer:
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, name: str, func=time.perf_counter):
        self.name = name
        self._func = func
        self.start = None
        self.total = None

    def __enter__(self):
        self.start = self._func()

    def __exit__(self, type=None, value=None, traceback=None):
        self.total = (self._func() - self.start) # Store the desired value.
        logger.info(f'{self.name}: {self.total:0.5f} s')
