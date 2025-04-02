from abc import abstractmethod
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Column
import pyspark
from typing import Iterator, Union
import re
import numpy as np
from delex.storage import StringIntHashMap
from delex.utils.traits import SparkDistributable
from delex.utils import CachedObjectKey
from pydantic.dataclasses import dataclass

class Tokenizer(SparkDistributable):

    @dataclass(frozen=True)
    class CacheKey(CachedObjectKey):
        index_col : str
        search_col : Union[str, None]
        tokenizer_type : str

    """
    Base class for all tokenizers

    Inherit from this class and implement the abstract `_tokenize` method to 
    create a new tokenizer that is compatible with set similarity predicates 
    such as Jaccard and Cosine
    """

    def __init__(self, use_freqs: bool=True):
        """
        Parameters
        ----------
        use_freqs : bool = True
            order the tokens by decreasing frequency, ordering by frequency is an important optimization for 
            indexing Jaccard, Cosine and other set similarity measures. This should be set to True unless there is a 
            good reason to set it to False. 
        """
        self._tok_to_id = None
        self._use_freqs = use_freqs
        self._nunique_tokens = None

    def __str__(self):
        return self.NAME

    def init(self):
        self._tok_to_id.init()

    def deinit(self):
        self._tok_to_id.deinit()

    def to_spark(self):
        self._tok_to_id.to_spark()

    @property
    def nunique_tokens(self):
        return self._nunique_tokens
    
    def tokenize_spark(self, input_col : Column):
        """
        return a column expression that gives the same output 
        as the tokenize method. Required for efficiency when building metadata for 
        certain methods
        """
        # spark treats whitespace differently than str.split
        # so make a udf to keep tokenization consistent
        @F.pandas_udf(T.ArrayType(T.IntegerType()))
        def t(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
            self._tok_to_id.init()
            for s in itr:
                yield s.apply(self.tokenize)
            self.deinit()

        return t(input_col)

    def tokenize_set_spark(self, input_col : Column):
        """
        return a column expression that gives the same output 
        as the tokenize method. Required for efficiency when building metadata for 
        certain methods
        """
        # spark treats whitespace differently than str.split
        # so make a udf to keep tokenization consistent
        @F.pandas_udf(T.ArrayType(T.IntegerType()))
        def t(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
            self._tok_to_id.init()
            for s in itr:
                yield s.apply(self.tokenize_set)
            self.deinit()

        return t(input_col)

    def tokenize(self, s: str) -> np.array:
        toks = self._tokenize(s)
        return self._tok_to_id[toks] if toks is not None else None
    
    @abstractmethod
    def _tokenize(self, s: str) -> list[str]:
        """
        convert the string into a BAG of tokens (tokens should not be deduped)

        This method should return None if s is None, if s produces no tokens 
        (i.e. len(s) == 0) this method should return an empty list, NOT None.
        """
    
    def tokenize_set(self, s: str) -> np.array:
        """
        tokenize the string and return a set or None if the tokenize returns None
        """
        toks = self._tokenize(s)
        if toks is None:
            return None
        else:
            # needed to get correct size when tokens are missing from the 
            # tok_to_id 
            toks = list(set(toks))
            tok_arr = self._tok_to_id[toks]
            tok_arr.sort()
            return tok_arr

    def __eq__(self, o):
        return isinstance(o, type(self)) and self.NAME == o.NAME

    def build(self, df: pyspark.sql.DataFrame, col: str):

        @F.pandas_udf(T.ArrayType(T.StringType()))
        def t(itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
            for s in itr:
                yield s.apply(self._tokenize)

        df = df.select(F.explode(t(col)).alias('TOK'))
        # order by increase freq for building jaccard, cosine, etc. indexes
        if self._use_freqs:
            df = df.groupby('TOK')\
                    .count()\
                    .orderBy(F.col('count').asc(), F.col('TOK').asc())\
                    .select('TOK')
        else:
            df = df.distinct()
    
        toks = df.toPandas()
        self._nunique_tokens = len(toks)
        self._tok_to_id = StringIntHashMap.build(toks['TOK'].values, np.arange(len(toks), dtype=np.int32), load_factor=.5)
        self.to_spark()

class StrippedWhiteSpaceTokenizer(Tokenizer):
    WHITESPACE_NORM = re.compile('\s+')
    RE = re.compile('[^a-z0-9 ]+')
    NAME='stripped_whitespace_tokens'

    def _tokenize(self, s):
        if isinstance(s, str):
            s = self.WHITESPACE_NORM.sub(' ', s).lower()
            s = self.RE.sub('', s)
            return s.split()
        else:
            return None

class ShingleTokenizer(Tokenizer):
    base_tokenize = StrippedWhiteSpaceTokenizer().tokenize

    def __init__(self, n):
        self._n = n
        self.NAME = f'{self._n}shingle_tokens'
        super().__init__()
    
    def _tokenize(self, s : str) -> list:
        single_toks = self.base_tokenize(s)
        if single_toks is None:
            return None

        if len(single_toks) < self._n:
            return []

        offsets = [0] + np.cumsum(list(map(len, single_toks))).tolist()
        slices = zip(offsets[:len(single_toks) - self._n], offsets[self._n:])
        combined = ''.join(single_toks)
        return [combined[s:e] for s,e in slices]

class WhiteSpaceTokenizer(Tokenizer):
    NAME='whitespace_tokens'
    def __init__(self):
        super().__init__()
        pass

    def _tokenize(self, s):
        return s.lower().split() if isinstance(s, str) else None


class NumericTokenizer(Tokenizer):

    NAME = 'num_tokens'
    def __init__(self):
        self._re = re.compile('[0-9]+')
        super().__init__()

    def _tokenize(self, s):
        return self._re.findall(s) if isinstance(s, str) else None

class AlphaNumericTokenizer(Tokenizer):
    #STOP_WORDS = set(stopwords.words('english'))
    NAME = 'alnum_tokens'
    def __init__(self):
        super().__init__()
        self._re = re.compile('[a-z0-9]+')


    def _tokenize(self, s):
        if not isinstance(s, str):
            return None
        else:
            return self._re.findall(s.lower())


class QGramTokenizer(Tokenizer):

    def __init__(self, q, use_freqs:bool = True):
        self._q = q
        self.NAME = f'{self._q}gram_tokens'
        super().__init__()

    def _tokenize(self, s : str) -> list:
        if not isinstance(s, str):
            return None
        if len(s) < self._q:
            return []
        s = s.lower()
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

class StrippedQGramTokenizer(Tokenizer):
    
    RE = re.compile('\\W+')
    def __init__(self, q, use_freqs:bool = True):
        self._q = q
        self.NAME = f'stripped_{self._q}gram_tokens'
        super().__init__()

    def _preproc(self, s : str) -> str:
        # strip all non-word chars
        return self.RE.sub('', s)

    def _tokenize(self, s : str) -> list:
        if not isinstance(s, str):
            return None

        s = self._preproc(s).lower()
        if len(s) < self._q:
            return []
        return [s[i:i+self._q] for i in range(len(s) - self._q + 1)]

