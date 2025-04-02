from xxhash import  xxh3_128_digest
from random import randint
import sys
from typing import Tuple


class HashFunction:
    """
    a simple wrapper class for the XXHash3
    """

    def __init__(self, seed=None):
        self._seed = seed if seed is not None else randint(0, 2**31)

    def hash_split(self, s: str, /) -> Tuple[int, int]:
        """
        hash `s` and return the 128 bits split between two ints
        """
        h = self.hash(s)
        return int.from_bytes(h[:8], sys.byteorder), int.from_bytes(h[8:], sys.byteorder)

    def hash(self, s: str) -> bytes:
        """
        hash `s` and return the 128 bits as bytes
        """
        return xxh3_128_digest(s, self._seed) 
