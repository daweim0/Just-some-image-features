import numpy as np


def compute_flow(left_features, right_features, mask, neighborhood_len = 6):
    #(np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    # neighborhood_len: side length of the square to search for corresponding features in
    mask = np.squeeze(mask)
    left_features = np.squeeze(left_features)
    right_features = np.squeeze(right_features)
    assert np.array_equal(left_features.shape, right_features.shape)
    assert np.array_equal(left_features.shape[0:2], mask.shape[0:2])
    flow_arr = np.zeros([left_features.shape[0], left_features.shape[1], 2], dtype=np.float32)
    for i in xrange(left_features.shape[0]):
        for j in xrange(left_features.shape[1]):
            if mask[i,j] == 0:
                best_index = np.zeros(2, dtype=np.int32)
                best_dist = float("inf")

                start_a = max(0, i - neighborhood_len/2)
                stop_a = min(left_features.shape[0], i + (neighborhood_len - neighborhood_len/2))
                start_b = max(0, j - neighborhood_len / 2)
                stop_b = min(left_features.shape[1], j + (neighborhood_len - neighborhood_len / 2))
                for a in xrange(start_a, stop_a):
                    for b in xrange(start_b, stop_b):
                        if mask[a, b] == 0:
                            current_dist = dist(left_features[i, j], right_features[a, b])
                            if current_dist < best_dist:
                                best_dist = current_dist
                                best_index[0] = a
                                best_index[1] = b

                flow_arr[i, j, 0] = i - best_index[0]
                flow_arr[i, j, 1] = j - best_index[1]

    return flow_arr


def dist(a, b):
    # (np.ndarray, ndarray) -> float
    # Calculates euclidean distance between a and b
    temp = np.power(a - b, 2)
    return np.sqrt(np.sum(temp))