# to speed up processing for images


import math
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel import parallel, prange, threadid
from libc.stdio cimport printf


def feature_similarity(l_features_obj, r_features_obj, flow_obj):
    return warp_c(l_features_obj, r_features_obj, flow_obj, l_features_obj.shape[0], l_features_obj.shape[1], l_features_obj.shape[2])


# bounds checking not turned off for debugging purpoces
cdef np.ndarray[np.float32_t, ndim=2] warp_c(np.ndarray[np.float32_t, ndim=3] l_features_obj,
                                             np.ndarray[np.float32_t, ndim=3] r_features_obj,
                                             np.ndarray[np.float32_t, ndim=3] flow_obj,
                                             int shape_0, int shape_1, int feature_depth):

    # warped = np.copy(image_left)
    # warped = np.copy(image_right)

    cdef float[:,:,:] l_features = l_features_obj
    cdef float[:,:,:] r_features = r_features_obj
    cdef float[:,:,:] flow = flow_obj
    cdef float[:,:] similar = np.zeros([shape_0, shape_1], dtype=np.float32)
    cdef float sum = 0.0
    # cdef int feature_depth = l_features_obj.shape[2]
    # cdef int shape_0 = l_features_obj.shape[0]
    # cdef int shape_1 = l_features_obj.shape[1]
    cdef int i, j, k, i_new, j_new

    for i in prange(shape_0, nogil=True, schedule='static', num_threads=16):
        for j in range(0, shape_1):
            i_new = i + <int> flow[i, j, 1]
            j_new = j + <int> flow[i, j, 0]
            if 0 <= i_new < shape_0:
                    if 0 <= j_new < shape_1:
                        sum = 0.0
                        for k in range(feature_depth):
                            sum = sum + (l_features[i, j, k] - r_features[i_new, j_new, k]) ** 2
                        similar[i, j] = sqrt(sum)

    return np.asarray(similar).copy()





    # similar = np.zeros(l_features.shape[0:2])
    # # warped = np.copy(image_left)
    # # warped = np.copy(image_right)
    # box_width = 14
    # for i in range(0, similar.shape[0], box_width):
    #     for j in range(0, similar.shape[1], box_width):
    #         for a in range(box_width / -2 + 1, box_width / 2, 2):
    #             if 0 <= i + a < similar.shape[0]:
    #                 for b in range(box_width / -2 + 1, box_width / 2, 2):
    #                     if 0 <= j + b < similar.shape[1]:
    #                         similar[i + a, j + b] = np.sqrt(
    #                             np.sum(np.power(l_features[i, j] - l_features[i + a, j + b], 2)))
    #                         similar[i+a+1, j+b] = similar[i + a, j + b]
    #                         similar[i+a, j+b+1] = similar[i + a, j + b]
    #                         similar[i+a+1, j+b+1] = similar[i + a, j + b]
    #         similar[i, j] = 1


    # cdef float[:,:,:] l_features = l_features_obj
    # cdef float[:,:,:] r_features = r_features_obj
    # cdef float[:,:,:] flow = flow_obj
    # warped = np.zeros([l_features.shape[0], l_features.shape[1], l_features.shape[2]])
    # # warped = np.copy(image_left)
    # # warped = np.copy(image_right)
    # for i in range(0, warped.shape[0], 2):
    #     for j in range(0, warped.shape[1], 2):
    #         i_new = i + int(flow[i, j, 1])
    #         j_new = j + int(flow[i, j, 0])
    #         if 0 <= i_new < warped.shape[0]:
    #                 if 0 <= j_new < warped.shape[1]:
    #                     warped[i, j] = np.sqrt(np.sum(np.power(l_features[i, j] - r_features[i_new, j_new], 2)))