import numba as nb
import numpy as np
from numba import njit
from numba.typed import List as TypedList
from numba.experimental import jitclass
from scipy import sparse
from delex.storage import PackedMemmapArrays 
from delex.utils.numba_functions import typed_list_to_array

UINT_16_MAX = np.iinfo(np.uint16).max

_spec = [
        ('nrow', nb.int32),
        ('ncol', nb.int32),
        ('data', nb.int16[:]),
        ('size', nb.int32[:]),
        ('indptr', nb.int32[:]),
        ('offset', nb.int32),
]
@jitclass(spec=_spec)
class SetSimIndexSlice:
    """
    a reference class for set similarity metrics, DO NOT USE THIS
    """

    def __init__(self, nrow, ncol, data, size, indptr, offset):
        if nrow > UINT_16_MAX+1:
            raise ValueError('slice too large')

        self.nrow = nrow
        self.ncol = ncol
        self.data = data
        self.size = size
        self.indptr = indptr
        self.offset = offset

    def _compute_overlap(self, olaps, indexes):
        for i in indexes:
            start  = self.indptr[i]
            end  = self.indptr[i+1]
            olaps[self.data[start:end]] += 1


    def _compute_jaccard(self, scores, indexes):
        self._compute_overlap(scores, indexes)
        for i in range(self.nrow):
            if scores[i] != 0:
                scores[i] /= (len(indexes) + self.size[i] - scores[i])

    def _compute_cosine(self, scores, indexes):
        self._compute_overlap(scores, indexes)
        for i in range(self.nrow):
            if scores[i] != 0:
                scores[i] /= np.sqrt(len(indexes) * self.size[i])

    def _compute_overlap_coeff(self, scores, indexes):
        self._compute_overlap(scores, indexes)
        for i in range(self.nrow):
            if scores[i] != 0:
                scores[i] /= min(len(indexes), self.size[i])

    def jaccard_threshold(self, indexes, thres, scores_out, indexes_out):
        scores = np.zeros(self.nrow, dtype=np.float32)
        self._compute_jaccard(scores, indexes)
        for i in range(self.nrow):
            if scores[i] >= thres:
                scores_out.append(scores[i])
                indexes_out.append(i + self.offset)

    def cosine_threshold(self, indexes, thres, scores_out, indexes_out):
        scores = np.zeros(self.nrow, dtype=np.float32)
        self._compute_cosine(scores, indexes)
        for i in range(self.nrow):
            if scores[i] >= thres:
                scores_out.append(scores[i])
                indexes_out.append(i + self.offset)

    def overlap_coeff_threshold(self, indexes, thres, scores_out, indexes_out):
        scores = np.zeros(self.nrow, dtype=np.float32)
        self._compute_overlap_coeff(scores, indexes)
        for i in range(self.nrow):
            if scores[i] >= thres:
                scores_out.append(scores[i])
                indexes_out.append(i + self.offset)


class SetSimIndex:

    def __init__(self):
        self.nrow = None
        self.ncol = None
        self.slices = None
        self._slice_size = None
        self._packed_arrs = None
        self._slc_shapes = None

    @classmethod
    def from_sparse_mat(cls, sparse_mat):
        obj = cls()
        if not isinstance(sparse_mat, (sparse.csr_matrix, sparse.csr_array)):
            raise TypeError
        obj.nrow, obj.ncol = sparse_mat.shape
        sparse_mat = sparse_mat.astype(np.float32)

        obj.slices = TypedList()

        obj._slice_size = 2**14
        offset = 0
        while offset < obj.nrow:
            start = offset
            end = min(obj.nrow, start + obj._slice_size)
            size = np.diff(sparse_mat.indptr[start:end+1])
            slc = sparse_mat[start:end].tocsc()

            data = np.empty(shape=len(slc.data), dtype=np.int16)
            data[:] = slc.indices 

            idx_slice = SetSimIndexSlice(
                        slc.shape[0],
                        slc.shape[1],
                        data,
                        size,
                        slc.indptr,
                        offset
                    )
            obj.slices.append(idx_slice)
            offset += obj._slice_size
        
        obj._packed_arrs = None
        obj._slc_shapes = None

        return obj


    def to_spark(self):
        self._slc_shapes = []
        arrs = []
        for slc in self.slices:
            arrs.append(slc.data)
            arrs.append(slc.size)
            arrs.append(slc.indptr)
            self._slc_shapes.append((slc.nrow, slc.ncol, slc.offset))

        self._packed_arrs = PackedMemmapArrays(arrs)
        self._packed_arrs.to_spark()
        self.slices = None

    def init(self):

        self.slices = TypedList()
        arr_itr = iter(self._packed_arrs.unpack())

        for nrows, ncols, offset in self._slc_shapes:
            data = next(arr_itr)
            size = next(arr_itr)
            indptr = next(arr_itr)

            self.slices.append( SetSimIndexSlice(
                        nrows,
                        ncols,
                        data, 
                        size,
                        indptr,
                        offset,
                    )
            )
    
    def jaccard_threshold(self, indexes, thres):
        return _jaccard_threshold(self.slices, indexes, thres)

    def overlap_coeff_threshold(self, indexes, thres):
        return _overlap_coeff_threshold(self.slices, indexes, thres)

    def cosine_threshold(self, indexes, thres):
        return _cosine_threshold(self.slices, indexes, thres)
            
njit_kwargs = {
        'nogil' : True,
        'fastmath' : True,
        'parallel' : False,
        'cache' : False
}

@njit( **njit_kwargs)
def _jaccard_threshold(slices, indexes, thres):
    scores_out = TypedList.empty_list(np.float32)
    indexes_out = TypedList.empty_list(np.int32)
    for s in slices:
        s.jaccard_threshold(indexes, thres, scores_out, indexes_out)
    
    return typed_list_to_array(scores_out), typed_list_to_array(indexes_out)

@njit( **njit_kwargs)
def _cosine_threshold(slices, indexes, thres):
    scores_out = TypedList.empty_list(np.float32)
    indexes_out = TypedList.empty_list(np.int32)
    for s in slices:
        s.cosine_threshold(indexes, thres, scores_out, indexes_out)
    
    return typed_list_to_array(scores_out), typed_list_to_array(indexes_out)

@njit( **njit_kwargs)
def _overlap_coeff_threshold(slices, indexes, thres):
    scores_out = TypedList.empty_list(np.float32)
    indexes_out = TypedList.empty_list(np.int32)
    for s in slices:
        s.overlap_coeff_threshold(indexes, thres, scores_out, indexes_out)
    
    return typed_list_to_array(scores_out), typed_list_to_array(indexes_out)
