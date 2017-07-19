# to speed up processing for images


import math
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import parallel, prange, threadid
from libc.stdio cimport printf


def warp():


# bounds checking not turned off for debugging purpoces
cdef np.ndarray[np.float32_t, ndim=3] warp(np.ndarray[np.float32_t, ndim=3] l_features_obj, np.ndarray[np.float32_t, ndim=3] flow_obj):
    cdef float[:,:,:] l_features = l_features_obj
    cdef float[:,:,:] flow = flow_obj
    warped = np.zeros([l_features.shape[0] / 4, l_features.shape[1] / 4, l_features.shape[2]])
    # warped = np.copy(image_left)
    # warped = np.copy(image_right)
    for i in range(0, l_features.shape[0], 4):
        for j in range(0, l_features.shape[1], 4):
                i_new = i + <int> flow[i, j, 1]
                j_new = j + <int> flow[i, j, 0]
                if 0 <= i_new < warped.shape[0] and 0 <= j_new < warped.shape[1]:
                    warped[i/4, j/4] = l_features[i_new, j_new] - r_features[i, j]