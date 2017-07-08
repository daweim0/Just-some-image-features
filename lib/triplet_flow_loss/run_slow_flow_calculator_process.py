import pyximport
# pyximport.install()
from slow_flow_calculator_cython import compute_flow
import subprocess
import numpy as np
import time
import sys
import os

def get_flow_parallel(left_features, right_features, mask, neighborhood_len_import=6, interpolate_after=False):
    random = str((time.clock() * 10000) % 1000 + 12345)
    np.save("temp/" + random + "neighborhood_len.npy", np.zeros([1], dtype=np.int32) + neighborhood_len_import)
    np.save("temp/" + random + "interpolate_after.npy", np.asarray([interpolate_after]))
    np.save("temp/" + random + "left_features.npy", left_features)
    np.save("temp/" + random + "right_features.npy", right_features)
    np.save("temp/" + random + "mask.npy", mask)
    # os.path.dirname(os.path.realpath(__file__))
    p = subprocess.Popen(['python', 'lib/triplet_flow_loss/run_slow_flow_calculator_process.py', random])
    p.wait()
    flow = np.load("temp/" + random + "flow.npy")
    subprocess.call("rm temp/" + random + "*", shell=True)
    return flow


if __name__ == "__main__":
    # time.sleep(0.5)
    # neighborhood_len_import = int(raw_input())
    # interpolate_after = bool(raw_input)
    random = sys.argv[1]
    # print random
    neighborhood_len_import = np.load("temp/" + random + "neighborhood_len.npy")[0]
    interpolate_after = np.load("temp/" + random + "interpolate_after.npy")[0]
    left_features = np.load("temp/" + random + "left_features.npy")
    right_features = np.load("temp/" + random + "right_features.npy")
    mask = np.load("temp/" + random + "mask.npy")
    flow = compute_flow(left_features, right_features, mask, neighborhood_len_import=neighborhood_len_import,
                 interpolate_after=interpolate_after)
    np.save("temp/" + random + "flow.npy", flow)
