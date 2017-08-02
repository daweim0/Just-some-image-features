import pyximport
# pyximport.install()
from slow_flow_calculator_cython import compute_flow
import subprocess
import numpy as np
import time
import sys
import os
import scipy.ndimage


def get_flow_parallel_pyramid(left_features, right_features, masks, neighborhood_len_import=6, interpolate_after=False):
    assert(len(left_features) == len(right_features) == len(masks))
    current_flow = np.zeros([left_features[0].shape[0], left_features[0].shape[1], 2], dtype=np.float32)
    #TODO: justify this as the correct neighborhood_len when using a pyramid
    search_range = list([left_features[0].shape[0] / float(left_features[-1].shape[0]) * neighborhood_len_import])
    for i in range(len(left_features) - 1):
        search_range.append(left_features[i+1].shape[0] / left_features[i].shape[0] * 2 + 4)

    output_flows = list()

    for i in range(len(left_features)):
        current_flow, feature_errors = get_flow_parallel(left_features[i], right_features[i], masks[i], neighborhood_len_import=int(search_range[i]),
                                         interpolate_after=interpolate_after, initialization=current_flow)
        output_flows.append(current_flow)
        if i < len(left_features) - 1:
            scales = list((left_features[i+1].shape[0] / left_features[i].shape[0], left_features[i+1].shape[1] / left_features[i].shape[1]))
            current_flow = scipy.ndimage.zoom(current_flow, scales + list([1]), order=1)
            current_flow *= scales
    return current_flow, feature_errors, output_flows


def get_flow_parallel(left_features, right_features, mask, neighborhood_len_import=6, interpolate_after=False, initialization=None):
    if initialization is None:
        initialization = np.zeros([left_features.shape[0], left_features.shape[1], 2], dtype=np.float32)
    random = str(time.clock() * os.getpid())
    np.save("temp/" + random + "neighborhood_len.npy", np.zeros([1], dtype=np.int32) + neighborhood_len_import)
    np.save("temp/" + random + "interpolate_after.npy", np.asarray([interpolate_after]))
    np.save("temp/" + random + "left_features.npy", left_features)
    np.save("temp/" + random + "right_features.npy", right_features)
    np.save("temp/" + random + "mask.npy", mask)
    np.save("temp/" + random + "initialization.npy", initialization)
    # os.path.dirname(os.path.realpath(__file__))
    p = subprocess.Popen(['python', 'lib/triplet_flow_loss/run_slow_flow_calculator_process.py', random])
    p.wait()
    # compute_stuff(random)
    flow = np.load("temp/" + random + "flow.npy")
    feature_errors = np.load("temp/" + random + "feature_errors.npy")
    subprocess.call("rm temp/" + random + "*", shell=True)
    return flow.copy(), feature_errors.copy()


def compute_stuff(random):
    # time.sleep(0.5)
    # neighborhood_len_import = int(raw_input())
    # interpolate_after = bool(raw_input)

    # print random
    neighborhood_len_import = np.load("temp/" + random + "neighborhood_len.npy")[0]
    interpolate_after = np.load("temp/" + random + "interpolate_after.npy")[0]
    left_features = np.load("temp/" + random + "left_features.npy")
    right_features = np.load("temp/" + random + "right_features.npy")
    mask = np.load("temp/" + random + "mask.npy")
    initialization = np.load("temp/" + random + "initialization.npy")
    output = compute_flow(left_features, right_features, mask, initialization, neighborhood_len_import=neighborhood_len_import,
                 interpolate_after=interpolate_after)
    flow = output[0]
    feature_errors = output[1]
    np.save("temp/" + random + "flow.npy", flow)
    np.save("temp/" + random + "feature_errors.npy", feature_errors)


if __name__ == "__main__":
    random = sys.argv[1]
    compute_stuff(random)