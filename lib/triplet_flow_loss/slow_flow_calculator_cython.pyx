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
from libc.stdio cimport printf


def compute_flow(left_features, right_features, mask, initialization, neighborhood_len_import=6, interpolate_after=False):
    if interpolate_after:
        interpolate = 1
    else:
        interpolate = 0
    feature_error_arr = np.zeros([left_features.shape[0], left_features.shape[1]], dtype=np.float32)
    assert np.array_equal(left_features.shape[:2], mask.shape)
    assert np.array_equal(left_features.shape, right_features.shape)
    assert np.array_equal(left_features.shape[:2], initialization.shape[:2])
    return compute_flow_helper(left_features.astype(np.float32), right_features.astype(np.float32), mask.astype(np.int32),
                               neighborhood_len_import, initialization.astype(np.float32), feature_error_arr, interpolate)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
@cython.cdivision(True)
cdef compute_flow_helper(np.ndarray[np.float32_t, ndim=3] left_features_obj,
                         np.ndarray[np.float32_t, ndim=3] right_features_obj, np.ndarray[np.int32_t, ndim=2] mask_obj,
                         int neighborhood_len_import, np.ndarray[np.float32_t, ndim=3] flow_arr_obj,
                         np.ndarray[np.float32_t, ndim=2] feature_error_obj, int interpolate_after):
    #(np.ndarray, np.ndarray, np.ndarray) -> np.ndarray, np.ndarray
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
    cdef float[:,:] feature_errors = feature_error_obj

    cdef float current_dist, best_dist, temp, temp2, best_index_u, best_index_v, best_a, best_b, x_average
    cdef int i, j, a, b, k, kk, feature_index, initial_i, initial_j

    cdef int u1, u2, v1, v2
    cdef float dist1, dist2

    cdef int best_x, best_y

    for i in prange(left_len_0, nogil=True, schedule='static', num_threads=23):
        for j in range(left_len_1):
            best_index_u = i
            best_index_v = j

            # set the starting distance to no be no movement. Otherwise the default will be large
            current_dist = 10**5  # just a really big number
            # if mask[i, j] == 0:

            initial_i = <int> flow_arr[i, j, 1] + i
            initial_j = <int> flow_arr[i, j, 0] + j
            current_dist = dist(left_features, right_features, i, j, initial_i, initial_j, feature_depth)
            best_dist = current_dist

            start_a = max_int(0, initial_i - neighborhood_len/2)
            stop_a = min_int(left_len_0, initial_i + (neighborhood_len - neighborhood_len/2))
            start_b = max_int(0, initial_j - neighborhood_len / 2)
            stop_b = min_int(left_len_1, initial_j + (<int> neighborhood_len - <int> neighborhood_len / 2))
            for a in range(start_a, stop_a):
                for b in range(start_b, stop_b):
                    # if mask[a, b] == 0:
                    current_dist = dist(left_features, right_features, i, j, a, b, feature_depth)

                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_index_u = a
                        best_index_v = b
            if best_dist > 2.0:
                best_index_u = -1


            if interpolate_after == 1 and best_index_u != -1:
                u1 = <int> best_index_u - 1
                if u1 < 0:
                    u1 = 0
                if u1 >= right_len_0 - 3:
                    u1 = right_len_0 - 3
                u2 = u1 + 2

                dist1 = dist(left_features, right_features, i, j, u1, <int> best_index_v, feature_depth)
                dist2 = dist(left_features, right_features, i, j, u2, <int> best_index_v, feature_depth)
                best_index_u = dist1 / (dist1 + dist2 + 0.00001) * u2 + dist2 / (dist1 + dist2 + 0.00001) * u1
                if not (u1 <= best_index_u and u2 >= best_index_u):
                    printf("i %3i, j, %3i, u1 %2i, u2 %2i, best_index_v %2.1f, dist1 %3.2f,\t dist2 %3.2f,\t best_index_u %3.1lf\n", i, j, u1, u2, best_index_v, dist1, dist2, best_index_u)

                v1 = <int> best_index_v - 1
                if v1 < 0:
                    v1 = 0
                if v1 >= right_len_1 - 3:
                    v1 = right_len_1 - 3
                v2 = v1 + 2

                dist1 = dist(left_features, right_features, i, j, <int> best_index_u, v1, feature_depth)
                dist2 = dist(left_features, right_features, i, j, <int> best_index_u, v2, feature_depth)
                best_index_v = (dist1 + 0.00001) / (dist1 + dist2 + 0.00002) * v2 + (dist2 + 0.00001) / (dist1 + dist2 + 0.00002) * v1
                if not (v1 <= best_index_v and v2 >= best_index_v):
                    printf("i %3i, j, %3i, v1 %2i, v2 %2i, best_index_u %2.1f, dist1 %3.2f,\t dist2 %3.2f,\t best_index_v %3.1lf\n", i, j, v1, v2, best_index_u, dist1, dist2, best_index_v)

            if best_index_u != -1:
                flow_arr[i, j, 1] = <float> (best_index_u - i)
                flow_arr[i, j, 0] = <float> (best_index_v - j)
                feature_errors[i, j] = <float> dist(left_features, right_features, i, j, <int> best_index_u, <int> best_index_v, feature_depth)
            else:
                flow_arr[i, j, 1] = 0.0
                flow_arr[i, j, 0] = 0.0

    return (np.asarray(flow_arr), np.asarray(feature_errors))


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


# # first axis
# if best_index_u == 0:
#     u1 = 0
#     u2 = 1
# elif best_index_u + 1 == left_len_0:
#     u1 = <int> best_index_u - 1
#     u2 = <int> best_index_u
# else:
#     dist1 = dist(left_features, right_features, i, j, <int> best_index_u - 1, j, feature_depth)
#     dist2 = dist(left_features, right_features, i, j, <int> best_index_u + 1, j, feature_depth)
#     if dist1 < dist2:
#         u1 = <int> best_index_u - 1
#         u2 = <int> best_index_u
#     else:
#         u1 = <int> best_index_u
#         u2 = <int> best_index_u + 1
#
# x_average = 0.0
# for feature_index in range(feature_depth):
#     x_average = x_average + ((left_features[i, j, feature_index] - right_features[u1, j, feature_index]) /
#                  (right_features[u2, j, feature_index] - right_features[u1, j, feature_index]))
# x_average = x_average / feature_depth
# if x_average < 0.0:
#     best_index_u = u1
# elif x_average > 1.0:
#     best_index_u = u2
# else:
#     best_index_u = best_index_u + x_average
#
# # second axis
# if best_index_v == 0:
#     v1 = 0
#     v2 = 1
# elif best_index_v + 1 == left_len_1:
#     v1 = <int> best_index_v - 1
#     v2 = <int> best_index_v
# else:
#     dist1 = dist(left_features, right_features, i, j, i, <int> best_index_v - 1, feature_depth)
#     dist2 = dist(left_features, right_features, i, j, i, <int> best_index_v + 1, feature_depth)
#     if dist1 < dist2:
#         v1 = <int> best_index_v - 1
#         v2 = <int> best_index_v
#     else:
#         v1 = <int> best_index_v
#         v2 = <int> best_index_v + 1
#
# x_average = 0.0
# for feature_index in range(feature_depth):
#     x_average = x_average + ((left_features[i, j, feature_index] - right_features[i, v1, feature_index]) /
#                  (right_features[i, v2, feature_index] - right_features[i, v1, feature_index]))
# x_average = x_average / feature_depth
# if x_average < 0.0:
#     best_index_v = v1
# elif x_average > 1.0:
#     best_index_v = v2
# else:
#     best_index_v = best_index_v + x_average