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
    np.save(random + "neighborhood_len.npy", np.zeros([1], dtype=np.int32) + neighborhood_len_import)
    np.save(random + "interpolate_after.npy", np.asarray([interpolate_after]))
    np.save(random + "left_features.npy", left_features)
    np.save(random + "right_features.npy", right_features)
    np.save(random + "mask.npy", mask)
    # os.path.dirname(os.path.realpath(__file__))
    p = subprocess.Popen(['python', 'lib/triplet_flow_loss/run_slow_flow_calculator_process.py', random])
    p.wait()
    flow = np.load(random + "flow.npy")
    subprocess.call("rm " + random + "*", shell=True)
    return flow



if __name__ == "__main__":
    # time.sleep(0.5)
    # neighborhood_len_import = int(raw_input())
    # interpolate_after = bool(raw_input)
    random = sys.argv[1]
    # print random
    neighborhood_len_import = np.load(random + "neighborhood_len.npy")[0]
    interpolate_after = np.load(random + "interpolate_after.npy")[0]
    left_features = np.load(random + "left_features.npy")
    right_features = np.load(random + "right_features.npy")
    mask = np.load(random + "mask.npy")
    flow = compute_flow(left_features, right_features, mask, neighborhood_len_import=neighborhood_len_import,
                 interpolate_after=interpolate_after)
    np.save(random + "flow.npy", flow)
