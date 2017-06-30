#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: cdivision=True

import math
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import parallel, prange, threadid

def compute_flow(left_features, right_features, mask, neighborhood_len_import=6, interpolate_after=False):
    if interpolate_after:
        interpolate = 1
    else:
        interpolate = 0
    flow_arr = np.zeros([left_features.shape[0], left_features.shape[1], 2], dtype=np.float32)
    assert np.array_equal(left_features.shape[:2], mask.shape)
    return compute_flow_helper(left_features.astype(np.float32), right_features.astype(np.float32), mask.astype(np.int32),
                               neighborhood_len_import, flow_arr, interpolate).copy()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] compute_flow_helper(np.ndarray[np.float32_t, ndim=3] left_features_obj, np.ndarray[np.float32_t, ndim=3] right_features_obj,
                 np.ndarray[np.int32_t, ndim=2] mask_obj, int neighborhood_len_import, np.ndarray[np.float32_t, ndim=3] flow_arr_obj,
                        int interpolate_after):
    #(np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    # neighborhood_len: side length of the square to search for corresponding features in
    # assert np.array_equal(left_features.shape, right_features.shape)
    # assert np.array_equal(left_features.shape[0:2], mask.shape[0:2])

    cdef int start_a, stop_a, start_b, stop_b
    cdef int left_len_0, left_len_1, right_len_0, right_len_1, feature_depth, neighborhood_len
    left_len_0 = <int> left_features_obj.shape[0]
    left_len_1 = <int> left_features_obj.shape[1]
    right_len_0 = <int> right_features_obj.shape[0]
    right_len_1 = <int> right_features_obj.shape[1]
    feature_depth = <int> left_features_obj.shape[2]
    neighborhood_len = <int> neighborhood_len_import

    cdef float[:,:,:] left_features = left_features_obj
    cdef float[:,:,:] right_features = right_features_obj
    cdef float[:,:,:] flow_arr = flow_arr_obj
    cdef int[:,:] mask = mask_obj

    cdef float current_dist, best_dist, temp, temp2, best_index_u, best_index_v, best_a, best_b
    cdef int i, j, a, b, k, kk

    # cdef np.ndarray[np.float32_t, ndim=2] flow_arr
    # cdef np.ndarray[np.float_t, ndims=3] left_features = np.array(left_features)
    # cdef np.ndarray[np.float_t, ndims=3] right_features = np.array(right_features)
    # cdef np.ndarray[np.float_t, ndims=2] mask = np.array(mask)

    # flow_arr = np.zeros([left_len_0, left_len_1, 2], dtype=np.float)

    cdef int best_x, best_y

    for i in prange(left_len_0, nogil=True, schedule='static', num_threads=25):
        for j in range(left_len_1):
            best_index_u = i
            best_index_v = j

            # set the starting distance to no be no movement. Otherwise the default will be large
            current_dist = 10**5  # just a really big number
            # if mask[i, j] == 0:
            current_dist = 0
            for k in range(feature_depth):
                temp = <float> left_features[i, j, k] - <float> right_features[i, j, k]
                current_dist += temp * temp
            current_dist = sqrt(current_dist)
            best_dist = current_dist

            start_a = max_int(0, i - neighborhood_len/2)
            stop_a = min_int(left_len_0, i + (neighborhood_len - neighborhood_len/2))
            start_b = max_int(0, j - neighborhood_len / 2)
            stop_b = min_int(left_len_1, j + (<int> neighborhood_len - <int> neighborhood_len / 2))
            for a in range(start_a, stop_a):
                for b in range(start_b, stop_b):
                    # if mask[a, b] == 0:
                    current_dist = dist(left_features, right_features, i, j, a, b, feature_depth)
                    # for kk in range(feature_depth):
                    #     temp = <float> left_features[i, j, kk] - <float> right_features[a, b, kk]
                    #     current_dist += temp ** 2
                    # current_dist = sqrt(current_dist)

                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_index_u = a
                        best_index_v = b

            if interpolate_after == 1:
                best_a = best_index_u
                best_b = best_index_v
                temp = 9999999
                if best_index_u < right_len_0 and dist(left_features, right_features, i, j, <int> best_index_u + 1, <int> best_index_v, feature_depth) < temp:
                    best_a = best_index_u + 1
                    best_b = best_index_v
                    temp = dist(left_features, right_features, i, j, <int> best_a, <int> best_b, feature_depth)
                if best_index_v + 1 < right_len_1 and dist(left_features, right_features, i, j, <int> best_index_u , <int> best_index_v + 1, feature_depth) < temp:
                    best_a = best_index_u
                    best_b = best_index_v + 1
                    temp = dist(left_features, right_features, i, j, <int> best_a, <int> best_b, feature_depth)
                if best_index_v - 1 >= 0 and dist(left_features, right_features, i, j, <int> best_index_u , <int> best_index_v - 1, feature_depth) < temp:
                    best_a = best_index_u
                    best_b = best_index_v - 1
                    temp = dist(left_features, right_features, i, j, <int> best_a, <int> best_b, feature_depth)
                if best_index_u - 1 >= 0 and dist(left_features, right_features, i, j, <int> best_index_u - 1, <int> best_index_v , feature_depth) < temp:
                    best_a = best_index_u - 1
                    best_b = best_index_v
                    temp = dist(left_features, right_features, i, j, <int> best_a, <int> best_b, feature_depth)
                temp2 = dist(left_features, right_features, i, j, <int> best_index_u, <int> best_index_v, feature_depth)

                best_index_u = temp / (temp + temp2) * best_a + temp2 / (temp + temp2) * best_index_u
                best_index_v = temp / (temp + temp2) * best_b + temp2 / (temp + temp2) * best_index_v


            flow_arr[i, j, 1] = <float> (best_index_u - i)
            flow_arr[i, j, 0] = <float> (best_index_v - j)
        # flow_arr[threadid(), j, 0] = <float> (9000 + threadid())

    return np.asarray(flow_arr)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
cdef inline float dist(float[:,:,:] left_features, float[:,:,:] right_features,
                int i, int j, int a, int b, int feature_depth) nogil:
    cdef float current_dist = 0
    cdef float temp = 0
    cdef int k
    for k in range(feature_depth):
        temp = <float> left_features[i, j, k] - <float> right_features[a, b, k]
        current_dist += temp ** 2
    return sqrt(current_dist)


cdef inline int max_int(int a, int b) nogil:
    if a < b:
        return b
    else:
        return a


cdef inline int min_int(int a, int b) nogil:
    if a > b:
        return b
    else:
        return a
