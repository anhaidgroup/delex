from pydantic.dataclasses import dataclass
import threading

@dataclass(frozen=True)
class CachedObjectKey:
    """
     A key for a cached object in the `BuildCache`
    """
    pass

class CacheItem:
    """
    A lockable item in the `BuildCache`. Essentially a 
    a pointer with a mutex to guard it for parallel builds
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._obj = None

    @property
    def obj(self):
        """
        the object (e.g. index, strings, tokenizer, etc.)
        """
        return self._obj

    @obj.setter
    def obj(self, o):
        """
        set the object for this cache item

        Raises
        ------
        RuntimeError
            if the obj has already been set
        """
        if self._obj is not None:
            raise RuntimeError('obj can only be set once')
        self._obj = o

    def __enter__(self):
        """
        lock the cache object
        """
        self._lock.acquire()

    def __exit__(self, type, value, traceback):
        """
        unlock the cache object
        """
        self._lock.release()

class BuildCache:
    """
    a cache of indexes, tokenizers, etc.
    """
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()

    def get(self, key: CachedObjectKey) -> CacheItem:
        """
        get the object associated with `key`. If `key` doesn't exist
        in the cache, adds a new CacheItem to cache and returns it


        Parameters
        ----------
        key : CachedObjectKey
            the key for the `CacheItem` being retrieved

        Returns
        -------
        `CacheItem` 
        """
        if not isinstance(key, CachedObjectKey):
            raise TypeError(type(key))
        with self._lock:
            if key not in self._cache:
                self._cache[key] = CacheItem()

        return self._cache[key]
