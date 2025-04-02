import numba as nb
import numpy as np
from numba.typed import List as TypedList
from numba.typed import Dict as TypedDict
from numba.experimental import jitclass
from delex.storage.span_map import create_span_map, span_map_get_key, span_map_entry_t
from delex.utils.numba_functions import sorted_set_overlap
import warnings
from numba.core.errors import NumbaTypeSafetyWarning

warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

INT_16_MAX = np.iinfo(np.int16).max
_int16_list_t = nb.types.ListType(nb.int16)

class FilteredSetSimIndexSlice:

    def __init__(self, nrow, thres, set_data, set_indptr, offset, data=None, span_map=None):
        # row oriented
        if nrow > INT_16_MAX+1:
            raise ValueError('slice too large')
        self.nrow = nrow
        self.thres = thres
        self.set_data = set_data
        self.set_indptr = set_indptr
        self.offset = offset
        
        
        #assert np.any(prefix_lens < self.size)

        if data is None or span_map is None:
            columns = TypedDict.empty(nb.int32, _int16_list_t)
            total = 0
            for i in range(self.nrow):
                sz = self.set_indptr[i+1] - self.set_indptr[i]
                prefix_len = self._prefix_len(sz, self.thres)

                total += prefix_len

                start = self.set_indptr[i]
                end = start + prefix_len
                for j in range(start, end):
                    token_id = self.set_data[j]
                    if token_id not in columns:
                        columns[token_id] = nb.typed.List.empty_list(nb.int16)
                    columns[token_id].append(i)

            self.data = np.empty(total, dtype=np.int16)

            non_empty_columns = len(columns)

            keys = np.empty(non_empty_columns, dtype=np.int32)
            offsets = np.zeros(non_empty_columns, dtype=np.int32)
            lengths = np.zeros(non_empty_columns, dtype=np.int16)
            i = 0 
            j = 0
            for token_id, col in columns.items():
                keys[j] = token_id
                offsets[j] = i
                lengths[j] = len(col)
                j += 1
                for idx in col:
                    self.data[i] = idx
                    i += 1

            self.span_map = create_span_map(keys, offsets, lengths)
        else:
            self.data = data
            self.span_map = span_map

    #@abstractmethod
    def _prefix_len(self, size, thres):
        pass
    
    #@abstractmethod
    def _size_bounds(self, query_size, thres):
        pass

    #@abstractmethod
    def _score(self, indexes, ids):
        pass

    def _prefix_filter(self, indexes, thres):
        prefix_len = self._prefix_len(len(indexes), thres)
        ids = set()
        for i in indexes[:prefix_len]:
            start, length = span_map_get_key(self.span_map, i)
            # no prefixes contain this token, skip
            if start < 0:
                continue
            end = start + length
            for j in range(start, end):
                ids.add(self.data[j])
        return ids

    def _size_filter(self, ids, indexes, thres):
        lower_bound, upper_bound, = self._size_bounds(len(indexes), thres)
        new_ids = TypedList()
        for i in ids:
            sz = self.set_indptr[i+1] - self.set_indptr[i]
            if lower_bound <= sz and sz <= upper_bound:
                new_ids.append(i)

        return new_ids
    
    def search(self, indexes, thres, scores_out, indexes_out):
        pids = self._prefix_filter(indexes, thres)
        ids = self._size_filter(pids, indexes, thres)
        # both the prefix filter and size filter are satisfied

        for i in ids:
            s = self._score(indexes, i)
            if s >= thres:
                scores_out.append(s)
                indexes_out.append(i + self.offset)


_SLICE_SPEC = [
        ('nrow', nb.int32),
        ('thres', nb.float32),
        ('data', nb.int16[:]),
        ('set_data', nb.int32[:]),
        ('size', nb.int32[:]),
        ('set_indptr', nb.int32[:]),
        ('span_map', nb.from_dtype(span_map_entry_t)[:]),
        ('offset', nb.int32),
]
@jitclass(_SLICE_SPEC)
class JaccardSetSimIndexSlice(FilteredSetSimIndexSlice):
        
    def _prefix_len(self, size, thres):
        # checked, this is correct
        sz = np.int32(size - np.ceil(size * thres) + 1)
        sz = max(sz, np.int32(0))
        sz = min(sz, size)
        return sz

    def _size_bounds(self, query_size, thres):
        lower_bound = np.ceil(query_size* thres)
        upper_bound = np.floor(query_size / thres)
        return np.int32(lower_bound), np.int32(upper_bound)

    def _score(self, indexes, i):
        start = self.set_indptr[i]
        end = self.set_indptr[i+1]
        olap = sorted_set_overlap(indexes, self.set_data[start:end])
        return olap / (len(indexes) + (end - start) - olap)


@jitclass(_SLICE_SPEC)
class CosineSetSimIndexSlice(FilteredSetSimIndexSlice):
        
    def _prefix_len(self, size, thres):
        sz = np.int32(size - np.ceil(thres**2 * size) + 1)
        sz = max(sz, np.int32(0))
        sz = min(sz, size)
        return sz

    def _size_bounds(self, query_size, thres):
        lower_bound = np.ceil(query_size * thres**2)
        upper_bound = np.floor(query_size / (thres**2))
        return np.int32(lower_bound), np.int32(upper_bound)

    def _score(self, indexes, i):
        start = self.set_indptr[i]
        end = self.set_indptr[i+1]
        olap = sorted_set_overlap(indexes, self.set_data[start:end])
        return olap / np.sqrt(len(indexes) * (end - start)) 
    

