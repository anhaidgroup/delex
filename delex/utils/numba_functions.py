import numba as nb
import numpy as np

njit_kwargs = {
        'nogil' : True,
        'fastmath' : True,
        'parallel' : False,
        'cache' : False
}
@nb.njit(**njit_kwargs)
def sorted_set_overlap(l_ind, r_ind, /):
    """
    compute the overlap between two sorted unique arrays

    Returns
    -------
    int 
    """
    l = 0
    r = 0
    s = 0

    while l < l_ind.size and r < r_ind.size:
        li = l_ind[l]
        ri = r_ind[r]
        if li == ri:
            s += 1
            l += 1
            r += 1
        elif li < ri:
            l += 1
        else:
            r += 1

    return s

@nb.njit( **njit_kwargs)
def typed_list_to_array(l):
    """
    covert a numba typed list to a numpy array
    """
    arr = np.empty(len(l), dtype=l._dtype)
    for i in range(len(l)):
        arr[i] = l[i]
    return arr
