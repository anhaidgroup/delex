"""
Tests for index.set_sim_index module.

This module tests set similarity indexing functionality.

**njit functions are not included in the pytest coverage. Also, we ignore the SetSimIndexSlice class. Our actual coverage is therefore: 100/101, missing line 101. 
"""
import pytest
import numpy as np
from scipy import sparse
from delex.index.set_sim_index import SetSimIndex


@pytest.mark.unit
class TestSetSimIndex:
    """Tests for SetSimIndex class."""

    def test_set_sim_index_init(self):
        """Test SetSimIndex initialization."""
        set_sim_index = SetSimIndex()
        assert set_sim_index is not None
        assert set_sim_index.nrow is None
        assert set_sim_index.ncol is None
        assert set_sim_index.slices is None
        assert set_sim_index._slice_size is None
        assert set_sim_index._packed_arrs is None
        assert set_sim_index._slc_shapes is None

    def test_set_sim_index_from_sparse_mat(self):
        """Test creating SetSimIndex from sparse matrix."""
        set_sim_index = SetSimIndex.from_sparse_mat(sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
        assert set_sim_index is not None
        assert set_sim_index.nrow == 3
        assert set_sim_index.ncol == 3
        assert set_sim_index.slices is not None
        assert len(set_sim_index.slices) == 1
        assert set_sim_index.slices[0].nrow == 3
        assert set_sim_index.slices[0].ncol == 3

    def test_set_sim_index_jaccard_threshold(self):
        """Test Jaccard similarity threshold queries."""
        set_sim_index = SetSimIndex.from_sparse_mat(sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).astype(np.float32))
        scores, ids = set_sim_index.jaccard_threshold(np.array([0, 1, 2]), 0.5)
        assert len(scores) == 2
        assert len(ids) == 2
        assert .66 < scores[0] < .67
        assert ids[0] == 0
        assert .66 < scores[1] < .67
        assert ids[1] == 2

    def test_set_sim_index_cosine_threshold(self):
        """Test Cosine similarity threshold queries."""
        set_sim_index = SetSimIndex.from_sparse_mat(sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).astype(np.float32))
        scores, ids = set_sim_index.cosine_threshold(np.array([0, 1, 2]), 0.5)
        assert len(scores) == 3
        assert len(ids) == 3
        assert 0.81 < scores[0] < 0.82
        assert ids[0] == 0
        assert 0.81 < scores[2] < 0.82
        assert ids[2] == 2

    def test_set_sim_index_overlap_coeff_threshold(self):
        """Test Overlap coefficient threshold queries."""
        set_sim_index = SetSimIndex.from_sparse_mat(sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).astype(np.float32))
        scores, ids = set_sim_index.overlap_coeff_threshold(np.array([0, 1, 2]), 0.5)
        assert len(scores) == 3
        assert len(ids) == 3
        assert scores[0] == 1.0
        assert ids[0] == 0
        assert scores[2] == 1.0
        assert ids[2] == 2

    def test_set_sim_index_to_spark(self):
        """Test converting SetSimIndex to Spark DataFrame and then re-initializing slices."""
        set_sim_index = SetSimIndex.from_sparse_mat(
            sparse.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).astype(np.float32)
        )
        set_sim_index.to_spark()
        assert set_sim_index._packed_arrs is not None
        assert set_sim_index._slc_shapes is not None
        assert len(set_sim_index._slc_shapes) == 1
        assert set_sim_index._slc_shapes[0] == (3, 3, 0)
        assert set_sim_index._packed_arrs.size_in_bytes() > 0
        assert set_sim_index.slices is None
        set_sim_index.init()
        assert set_sim_index.slices is not None
        assert len(set_sim_index.slices) == 1
        slc = set_sim_index.slices[0]
        assert slc.nrow == 3
        assert slc.ncol == 3
        assert slc.offset == 0