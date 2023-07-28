from pathlib import Path
import contextlib
import functools
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle
import shutil
from typing import Optional, Literal

import numpy as np
import torch


logging.getLogger(__name__)

__all__ = [
    'Cache',
    'TorchCache',
    'NpzCache',
    'PklCache',
    'torchcache',
]

def mkdir(root_path, default: Optional[Literal['o', 'l', 'a']] = None):
    """
    root_path should be a pathlib.Path
    """
    if root_path.exists():
        # Overwrite?
        if default is not None:
            opt = default
        else:
            opt = input(
                f'{root_path} exists. [O]verwrite/[L]eave/[A]bort? [o/l/a] '
            ).lower()
        if opt == 'o':
            print('Overwriting.')
            shutil.rmtree(root_path)
        elif opt == 'l':
            print('Continuing.')
        else:
            print('Aborting.')
            sys.exit()
    Path(root_path).mkdir(parents=True, exist_ok=True)

class Cache:
    """Simple cache that uses filenames to tell if something has been stored.
    Mostly suited for single runs/debugging, not long-term results storage

    Beware: No checking to see if the cache really needs to be updated!
    """
    @classmethod
    def cache(cls, cachefile: Path, force_reload: bool = False):
        mkdir(cachefile.parent, default='l')
        def decorator(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                if cachefile.is_file() and not force_reload:
                    logging.info(f'Loading cached result of {fn.__name__}')
                    out = cls._load(cachefile)
                else:
                    out = fn(*args, **kwargs)
                    cls._save(out, cachefile)
                return out
            return wrapped
        return decorator

    @staticmethod
    def _save(data, cachefile: Path):
        return NotImplemented

    @staticmethod
    def _load(cachefile: Path):
        return NotImplemented


class TorchCache(Cache):
    """Use mostly for saving torch tensors"""
    @staticmethod
    def _save(data, cachefile: Path):
        return torch.save(data, cachefile)

    @staticmethod
    def _load(cachefile: Path):
        return torch.load(cachefile)


class NpzCache(Cache):
    """Use mostly for saving arrays, dictionaries aren't as good"""
    @staticmethod
    def _save(data, cachefile: Path):
        return np.savez(cachefile, data=data)

    @staticmethod
    def _load(cachefile: Path):
        data = np.load(cachefile, allow_pickle=True)['data']
        breakpoint()
        return data

class PklCache(Cache):
    """Use for saving generic python objects"""
    @staticmethod
    def _save(data, cachefile: Path):
        with open(cachefile, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def _load(cachefile: Path):
        with open(cachefile, 'rb') as f:
            return pickle.load(f)

def torchcache(cachefile: Path, force_reload: bool = False):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if cachefile.is_file() and not force_reload:
                logging.info(f'Loading cached result of {fn.__name__}')
                out = torch.load(cachefile)
            else:
                out = fn(*args, **kwargs)
                torch.save(out, cachefile)
            return out
        return wrapped
    return decorator

def test_npzcache():
    from easydict import EasyDict
    data = {
        'a': np.random.randn(3, 3),
        'x': [np.random.randn(1), np.random.randn(2)],
        'y': EasyDict({
            'z': np.random.randn(4),
        }),
    }

    @PklCache.cache(Path('test.pkl'))
    def getdata():
        return data
    data = getdata()


if __name__ == '__main__':
    test_npzcache()
