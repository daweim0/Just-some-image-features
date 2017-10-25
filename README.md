# Some In-progress Possibly Overkill Dense Image Features

This code trains dense image features then computes optical flow and image correspondence using them (dense features are a primary goal, optical flow is just a metric). It was built out of Yu Xiang's code base [DA-RNN](https://github.com/yuxng/DA-RNN).

This project is by no means done. I started it while spending the summer after freshman year in the University of Washington's Robotics and State Estimation Lab and stopped working on it when classes started. Hopefully I'll come back to it some day.

Testing data and a trained model can be downloaded [here](https://drive.google.com/file/d/0B0ANYo6Rw6Jxci04WC1lZWFlOG8/view?usp=sharing).

### Setup Instructions

1. Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup). We use Python 2.7, a virtualenv is recommended. Make sure to install CUDA and CUDNN along the way.

2. Clone the repo
```Shell
git clone https://github.com/daweim0/Just-some-image-features.git
```

3. Compile all the cython and c files. This consists of running
```Shell
cd lib/triplet_flow_loss && ./build_cython.sh && cd ../..
cd lib/triplet_flow_loss && ./make_with_cuda.sh  && cd ../..  (or make_cpu.sh if you want debugging symbols)
cd lib/gt_lov_correspondence_layer && ./build_cython.sh && cd ../..
cd lib/gt_lov_synthetic_layer && ./build_cython.sh && cd ../..
```

4. Download the training data from here

5. Run one of the scripts in /experiments/scripts. They should be run from the repo's root directory. For example, to train features on the synthetic LOV dataset using GPU 0:
    ```Shell
    ./experiments/scripts/lov_features_synthetic.sh 0
    ```

6. Once a model is trained it can be tested like so
    ```Shell
    ./experiments/scripts/lov_features_synthetic_60.sh 0 output/lov_synthetic_features_fpn_2x/batch_size_3_loss_L2_optimizer_ADAM_network_net_labeled_fpn_fixed_2017-8-14-17-41-39/vgg16_flow_lov_features_iter_26000.ckpt
    ```
  (or you can use your own checkpoints)

### Sample Output

The following images show optical flow calculated by a nearest neighbor search on image feature maps, individually taking every pixel in the left image and looking for the most similar pixel in the right image. The visualization is direct output from the nearest neighbor search, **no post-processing or smoothing** has been done (since the point is to demonstrate how discriminative the features are, not fancy optical flow algorithms).

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_21.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_28.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_57.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_59.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_62.png)

![Some output](https://raw.githubusercontent.com/daweim0/Just-some-image-features/more_comments/plot_64.png)

### License

This code is released under the MIT License (refer to the LICENSE file for details).
