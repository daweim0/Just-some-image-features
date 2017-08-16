# Some overkill image features

This code trains dense image features then computes optical flow using them (dense features are a primary goal, optical flow is just a metric).

This hopefully won't become the dumping ground of a dead thing.

### Installation Instructions

1. Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup). We use Python 2.7, a virtualenv is recommended. 

2. Install [CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#abstract)

3. Compile all the cython and c files. This consists of running
    ```Shell
  lib/triplet_flow_loss/build_cython.sh
  lib/triplet_flow_loss/make_with_cuda.sh (or make_cpu.sh if you want debugging symbols)
  lib/gt_lov_correspondence_layer/build_cython.sh
  lib/gt_lov_synthetic_layer/build_cython.sh
    ```

4. Run one of the scripts in /experiments/scripts. They should be run from the repo's root directory. For example, to train features on the synthetic LOV dataset using GPU 0:
    ```Shell
    ./experiments/scripts/lov_features_synthetic.sh 0
    ```

5. Once a model is trained it can be tested like so
    ```Shell
    ./experiments/scripts/lov_features_synthetic_60.sh 0 output/lov_synthetic_features_fpn_2x/batch_size_3_loss_L2_optimizer_ADAM_network_net_labeled_fpn_fixed_2017-8-14-17-41-39/vgg16_flow_lov_features_iter_26000.ckpt
    ```
  (or you can use your own trained network)

### License

This code is released under the MIT License (refer to the LICENSE file for details).
